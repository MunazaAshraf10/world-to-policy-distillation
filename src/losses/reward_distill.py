import torch


def reward_loss(student_reward: torch.Tensor, teacher_reward: torch.Tensor) -> torch.Tensor:
    """World reward distillation L_reward = ||r(tau^S) - r(tau*_T)||_2 (Eq. 16), both [B].

    Both rewards come from the same frozen reward model in the same predicted world, so the
    gradient reaches the student through its trajectory.
    """
    return (student_reward - teacher_reward).abs().mean()
