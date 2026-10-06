import torch


def policy_loss(student_query: torch.Tensor, teacher_query: torch.Tensor) -> torch.Tensor:
    """Policy distillation L_policy = ||Q^S - Q^T||_2 (Eq. 15), averaged over the batch.

    student_query: [B, D] student plan query mapped into the teacher's feature space.
    teacher_query: [B, D] refined teacher query of the reward selected mode.
    """
    return torch.linalg.vector_norm(student_query - teacher_query, dim=-1).mean()
