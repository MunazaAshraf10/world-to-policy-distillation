import torch


def total_loss(
    terms: dict[str, torch.Tensor], weights: dict[str, float]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of named loss terms; returns the scalar and unweighted values for logging."""
    unknown = terms.keys() - weights.keys()
    if unknown:
        raise KeyError(f"No weight configured for loss terms {sorted(unknown)}")
    total = sum(weights[name] * value for name, value in terms.items())
    logs = {name: value.detach().item() for name, value in terms.items()}
    logs["total"] = total.detach().item()
    return total, logs
