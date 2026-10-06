from dataclasses import dataclass

import torch

FREE = 17
DRIVABLE = 11
OBJECT_CLASSES = tuple(range(11))
OFFROAD_CLASSES = (12, 13, 14, 15, 16)


@dataclass(frozen=True, slots=True)
class Rollout:
    """OccWorld evaluation protocol: five observed frames, six predicted at 2 Hz."""

    history: int = 5
    future: int = 6
    window: int = 12
    grid: tuple[int, int, int] = (200, 200, 16)
    latent: tuple[int, int] = (50, 50)
    classes: int = 18
    codebook_size: int = 512
    step_seconds: float = 0.5


@dataclass(frozen=True, slots=True)
class WorldPrediction:
    """Frozen world model output. Ego values are per-step displacements in the LiDAR frame."""

    codes: torch.Tensor
    occupancy: torch.Tensor
    ego_disp: torch.Tensor
    ego_disp_modes: torch.Tensor

    def validate(self, rollout: Rollout) -> None:
        bs = self.codes.shape[0]
        expected = {
            "codes": (bs, rollout.future, *rollout.latent),
            "occupancy": (bs, rollout.future, *rollout.grid),
            "ego_disp": (bs, rollout.future, 2),
            "ego_disp_modes": (bs, rollout.future, 3, 2),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(tensor.shape)}")
            if tensor.requires_grad:
                raise ValueError(f"Frozen prediction {name} requires gradients")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains nonfinite values")
        if self.codes.min() < 0 or self.codes.max() >= rollout.codebook_size:
            raise ValueError("Predicted codes fall outside the codebook")
        if self.occupancy.min() < 0 or self.occupancy.max() >= rollout.classes:
            raise ValueError("Predicted occupancy labels fall outside the class range")
