import subprocess
import sys

import pytest
import torch
from torch import nn

from src.data.cache import pack_bev
from src.losses.policy_distill import policy_loss
from src.losses.total import total_loss
from src.policy.student import Student, StudentConfig
from src.policy.teacher import Teacher, TeacherConfig, trajectory_anchors
from src.reward.model import RewardConfig, RewardModel, final_reward
from src.reward.targets import SimConfig, simulation_targets
from src.training.steps import teacher_step, wpt_step

B, K, T = 2, 3, 6


@pytest.fixture(scope="module")
def models() -> dict[str, nn.Module]:
    torch.manual_seed(0)
    codebook = torch.randn(512, 16)
    return {
        "student": Student(StudentConfig(dim=32, widths=(8, 16, 32), layers=1, heads=2)),
        "teacher": Teacher(TeacherConfig(dim=32, layers=1, heads=2, modes=K), codebook),
        "reward": RewardModel(RewardConfig(dim=32, layers=1, heads=2), codebook),
        "projector": nn.Linear(32, 32),
    }


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    step = torch.tensor([0.0, 2.0])
    masks = torch.zeros(B, T, 2, 200, 200, dtype=torch.bool)
    masks[:, :, 1] = True
    return {
        "occ": torch.randint(0, 18, (B, 5, 200, 200, 16), dtype=torch.uint8),
        "codes": torch.randint(0, 512, (B, T, 50, 50)),
        "command": torch.eye(3)[[0, 2]],
        "ego_hist": torch.zeros(B, 2, 2),
        "target": step.expand(B, T, 2).clone(),
        "target_mask": torch.ones(B, T),
        "bev": pack_bev(masks),
        "ego_disp": step.expand(B, T, 2).clone(),
        "lidar2ego": torch.tensor([[0.0, 1.0, 0.94], [-1.0, 0.0, 0.0]]).expand(B, 2, 3),
    }


def test_output_shapes(models: dict[str, nn.Module], batch: dict[str, torch.Tensor]) -> None:
    student = models["student"](batch["occ"], batch["command"], batch["ego_hist"])
    assert student.query.shape == (B, 32) and student.traj.shape == (B, T, 2)
    teacher = models["teacher"](batch["codes"], batch["command"], batch["ego_hist"])
    assert teacher.query.shape == (B, K, 32) and teacher.traj_set.shape == (B, K, T, 2)
    scores = models["reward"](batch["codes"], batch["ego_disp"], teacher.traj_set)
    assert scores.im_logit.shape == (B, K) and scores.sim_logit.shape == (B, K, 5)
    assert torch.isfinite(final_reward(scores, (1.0, 1.0, 1.0, 1.0), 1e-6)).all()


def test_anchors_recover_separated_trajectories() -> None:
    torch.manual_seed(0)
    ends = torch.tensor([[0.0, 0.0], [-8.0, 20.0], [8.0, 20.0]])
    steps = torch.linspace(1 / T, 1, T)[:, None]
    traj = (ends[:, None] * steps).repeat_interleave(50, dim=0) + 0.1 * torch.randn(150, T, 2)
    anchors = trajectory_anchors(traj, k=3)
    found = anchors[:, -1][anchors[:, -1, 0].argsort()]
    assert torch.allclose(found, ends[ends[:, 0].argsort()], atol=0.2)


def test_teacher_candidates_stay_distinct(batch: dict[str, torch.Tensor]) -> None:
    torch.manual_seed(0)
    teacher = Teacher(TeacherConfig(dim=32, layers=1, heads=2, modes=K), torch.randn(512, 16))
    teacher.anchors.copy_(torch.linspace(-5.0, 5.0, K)[:, None, None].expand(K, T, 2))
    traj_set = teacher(batch["codes"], batch["command"], batch["ego_hist"]).traj_set
    spread = (traj_set[:, :, None] - traj_set[:, None]).norm(dim=-1).mean(-1)
    assert spread[:, ~torch.eye(K, dtype=torch.bool)].min() > 1.0


def test_reward_reads_predicted_ego_path(
    models: dict[str, nn.Module], batch: dict[str, torch.Tensor]
) -> None:
    traj_set = batch["target"].cumsum(1)[:, None].expand(B, K, T, 2)
    reward = models["reward"].eval()
    with torch.no_grad():
        near = reward(batch["codes"], batch["ego_disp"], traj_set).im_logit
        far = reward(batch["codes"], batch["ego_disp"] + 3.0, traj_set).im_logit
    assert not torch.allclose(near, far)


def test_policy_loss_vanishes_for_identical_queries() -> None:
    query = torch.randn(4, 32)
    assert policy_loss(query, query.clone()).item() == 0.0
    assert policy_loss(query, query + 1.0).item() > 0.0


def test_simulation_targets_follow_predicted_world(batch: dict[str, torch.Tensor]) -> None:
    straight = batch["target"].cumsum(1)[:, None]
    masks = torch.zeros(B, T, 2, 200, 200, dtype=torch.bool)
    masks[:, :, 1] = True
    clear = simulation_targets(straight, masks, batch["ego_disp"], batch["lidar2ego"], SimConfig())
    assert torch.equal(clear[..., :2], torch.ones(B, 1, 2))
    # The ego stays at the center of each predicted frame, so blocking it causes a collision.
    masks[:, :, 0, 98:102, 98:102] = True
    blocked = simulation_targets(
        straight, masks, batch["ego_disp"], batch["lidar2ego"], SimConfig()
    )
    assert torch.equal(blocked[..., 0], torch.zeros(B, 1))


def test_teacher_step_trains_teacher_and_reward(
    models: dict[str, nn.Module], batch: dict[str, torch.Tensor]
) -> None:
    for name in ("teacher", "reward"):
        models[name].requires_grad_(True).zero_grad(set_to_none=True)
    terms = teacher_step(batch, models["teacher"], models["reward"], SimConfig())
    loss, logs = total_loss(terms, {"wta": 1.0, "im": 1.0, "sim": 1.0})
    loss.backward()
    assert all(torch.isfinite(torch.tensor(v)) for v in logs.values())
    assert models["teacher"].head.mlp[-1].weight.grad is not None
    assert models["reward"].im_head.weight.grad is not None


def test_wpt_step_gradient_boundaries(
    models: dict[str, nn.Module], batch: dict[str, torch.Tensor]
) -> None:
    for name, model in models.items():
        frozen = name in ("teacher", "reward")
        model.requires_grad_(not frozen).zero_grad(set_to_none=True)
        model.train(not frozen)
    terms = wpt_step(
        batch, models["student"], models["teacher"], models["reward"], models["projector"]
    )
    loss, logs = total_loss(terms, {"plan": 1.0, "policy": 1.0, "reward": 0.5})
    loss.backward()
    assert set(logs) == {"plan", "policy", "reward", "total"}
    assert all(torch.isfinite(v) for v in terms.values())
    for name in ("teacher", "reward"):
        assert all(p.grad is None for p in models[name].parameters())
    for name in ("student", "projector"):
        assert all(p.grad is not None for p in models[name].parameters() if p.requires_grad)


def test_student_inference_needs_no_world_model() -> None:
    code = (
        "import sys, torch\n"
        "from src.policy.student import Student, StudentConfig\n"
        "m = Student(StudentConfig(dim=32, widths=(8, 16, 32), layers=1, heads=2)).eval()\n"
        "with torch.no_grad():\n"
        "    m(torch.zeros(1, 5, 200, 200, 16, dtype=torch.uint8), torch.eye(3)[:1], "
        "torch.zeros(1, 2, 2))\n"
        "loaded = [n for n in sys.modules if n.startswith(('src.world', 'src.reward', "
        "'src.policy.teacher', 'mmengine'))]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
