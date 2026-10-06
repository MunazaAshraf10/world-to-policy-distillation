import torch
from torch import nn

from src.data.cache import unpack_bev
from src.losses.planning import plan_loss, wta_loss
from src.losses.policy_distill import policy_loss
from src.losses.reward import imitation_loss, simulation_loss
from src.losses.reward_distill import reward_loss
from src.policy.student import Student
from src.policy.teacher import Teacher, TeacherOutput
from src.reward.model import RewardModel, final_reward
from src.reward.targets import SimConfig, imitation_target, simulation_targets

type Batch = dict[str, torch.Tensor]


def expert_positions(batch: Batch) -> torch.Tensor:
    """Per-step expert displacements accumulated into positions [B, T, 2]."""
    return batch["target"].cumsum(dim=1)


def student_step(batch: Batch, student: Student) -> dict[str, torch.Tensor]:
    out = student(batch["occ"], batch["command"], batch["ego_hist"])
    return {"plan": plan_loss(out.traj, expert_positions(batch), batch["target_mask"])}


def teacher_step(
    batch: Batch, teacher: Teacher, reward: RewardModel, sim: SimConfig
) -> dict[str, torch.Tensor]:
    """Teacher imitation plus reward model supervision on the teacher's own candidates.

    Candidates are detached before scoring so the reward losses train the reward model only;
    the teacher is shaped by its imitation loss as in Baseline-T.
    """
    target = expert_positions(batch)
    out = teacher(batch["codes"], batch["command"], batch["ego_hist"])
    traj_set = out.traj_set.detach()
    with torch.no_grad(), torch.autocast(traj_set.device.type, enabled=False):
        im_target = imitation_target(
            traj_set,
            target,
            batch["target_mask"],
            reward.cfg.im_temperature,
            literal=reward.cfg.im_target == "literal",
        )
        sim_target = simulation_targets(
            traj_set,
            unpack_bev(batch["bev"]),
            batch["ego_disp"],
            batch["lidar2ego"],
            sim,
        )
    scores = reward(batch["codes"], traj_set)
    return {
        "wta": wta_loss(out.traj_set, target, batch["target_mask"]),
        "im": imitation_loss(scores.im_logit, im_target),
        "sim": simulation_loss(scores.sim_logit, sim_target),
    }


def select_best(
    teacher: Teacher, reward: RewardModel, batch: Batch
) -> tuple[TeacherOutput, torch.Tensor]:
    """Teacher outputs with the reward selected mode tau*_T = argmax r_final (Eq. 14)."""
    out = teacher(batch["codes"], batch["command"], batch["ego_hist"])
    score = final_reward(reward(batch["codes"], out.traj_set), reward.cfg.alpha, reward.cfg.eps)
    return out, score.argmax(dim=-1)


def wpt_step(
    batch: Batch,
    student: Student,
    teacher: Teacher,
    reward: RewardModel,
    projector: nn.Module,
) -> dict[str, torch.Tensor]:
    """Student training with policy distillation (Eq. 15) and world reward distillation (Eq. 16).

    The teacher and reward model are frozen. The student trajectory is appended to the teacher
    candidates and scored in the same predicted world, so the imitation softmax of Eq. 14 is
    normalized over a common set rather than a singleton.
    """
    with torch.no_grad():
        plan, best = select_best(teacher, reward, batch)
    rows = torch.arange(best.shape[0], device=best.device)
    out = student(batch["occ"], batch["command"], batch["ego_hist"])
    candidates = torch.cat([plan.traj_set, out.traj[:, None]], dim=1)
    score = final_reward(reward(batch["codes"], candidates), reward.cfg.alpha, reward.cfg.eps)
    return {
        "plan": plan_loss(out.traj, expert_positions(batch), batch["target_mask"]),
        "policy": policy_loss(projector(out.query), plan.query[rows, best]),
        "reward": reward_loss(score[:, -1], score[rows, best].detach()),
    }
