import torch
import torch.nn as nn


class StablemaxCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100, reduction: str = "none", epsilon: float = 1e-30):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.epsilon = epsilon

    def _stable_transform(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.where(
            logits < 0,
            1 / (1 - logits + self.epsilon),
            logits + 1,
        )

    def _log_stablemax(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        stable_values = self._stable_transform(logits)
        return torch.log(stable_values / torch.sum(stable_values, dim=dim, keepdim=True))

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_probabilities = self._log_stablemax(logits.to(torch.float64), dim=-1)

        if valid_mask is None:
            valid_mask = labels != self.ignore_index

        safe_label_indices = torch.where(valid_mask, labels, 0)
        target_log_probs = torch.gather(
            log_probabilities,
            index=safe_label_indices.to(torch.long).unsqueeze(-1),
            dim=-1,
        ).squeeze(-1)

        per_token_loss = -torch.where(valid_mask, target_log_probs, 0.0)

        if self.reduction == "mean":
            if valid_mask.any():
                return per_token_loss[valid_mask].mean()
            return per_token_loss.sum()
        if self.reduction == "sum":
            return per_token_loss[valid_mask].sum() if valid_mask.any() else per_token_loss.sum()
            
        return per_token_loss
