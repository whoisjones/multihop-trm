import copy

def _unwrap_module(module):
    """Unwrap DataParallel, DistributedDataParallel, or Accelerator-prepared model."""
    while hasattr(module, 'module'):
        module = module.module
    return module

class EMAHelper(object):
    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        module = _unwrap_module(module)
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        module = _unwrap_module(module)
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        """Copy EMA weights into the module (in-place)."""
        module = _unwrap_module(module)
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def store(self, module):
        """Save current model params (e.g. before swapping in EMA for eval)."""
        module = _unwrap_module(module)
        self._stored = {}
        for name, param in module.named_parameters():
            if param.requires_grad:
                self._stored[name] = param.data.clone()

    def restore(self, module):
        """Restore previously stored params (e.g. after eval with EMA)."""
        module = _unwrap_module(module)
        for name, param in module.named_parameters():
            if param.requires_grad and name in self._stored:
                param.data.copy_(self._stored[name])

    def ema_copy(self, module):
        """Return a copy of the module with EMA weights applied."""
        module_copy = copy.deepcopy(_unwrap_module(module))
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict