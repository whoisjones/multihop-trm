import torch
from torch.optim import Optimizer

def _sparse_emb_signsgd_dist(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    lr: float,
    weight_decay: float,
    world_size: int
) -> None:
    N, D = local_weights_grad.shape
    
    all_weights_grad = local_weights_grad
    all_ids = local_ids

    if world_size > 1:
        all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * N, dtype=local_ids.dtype, device=local_ids.device)
    
        torch.distributed.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        torch.distributed.all_gather_into_tensor(all_ids, local_ids)

    grad_ids, inv = all_ids.unique(return_inverse=True)

    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    p = weights[grad_ids]

    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

    weights[grad_ids] = p

class SparseEmbeddingSignSGD(Optimizer):
    def __init__(
        self,
        params,
        world_size: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            world_size=world_size
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure = None):  # type: ignore
        for group in self.param_groups:
            local_weights_grad = None
            local_ids = None
            weights = None
            
            assert len(group["params"]) == 3
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                else:
                    assert False
                
            assert local_ids is not None
            assert weights is not None
        
            if local_weights_grad is not None:
                _sparse_emb_signsgd_dist(
                    local_weights_grad,
                    local_ids,
                    weights,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    world_size=group["world_size"]
                )
