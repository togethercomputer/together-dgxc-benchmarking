"""
Compatibility wrapper for running 25.09-generated NeMo configs in 26.02 container.

Fixes:
- enable_cuda_graph (deprecated) -> cuda_graph_impl (new API in 26.02 Megatron-LM)
- no_weight_decay_cond kwarg rejected by older Megatron-Core get_megatron_optimizer()

Usage: python /path/to/compat_runner.py -n <name> <config_path>
"""
import sys
import types
import inspect
import warnings

# Stub the missing tensorstore submodule BEFORE anything imports dist_ckpt_io.
# The 26.02 container's megatron.core.dist_checkpointing.strategies doesn't include
# tensorstore, causing an ImportError that crashes the job during strategy teardown.
# Since checkpointing is disabled for benchmarking, a no-op stub is safe.
try:
    from megatron.core.dist_checkpointing.strategies import tensorstore  # noqa: F401
except ImportError:
    _stub = types.ModuleType('megatron.core.dist_checkpointing.strategies.tensorstore')
    sys.modules['megatron.core.dist_checkpointing.strategies.tensorstore'] = _stub
    import megatron.core.dist_checkpointing.strategies as _strats
    _strats.tensorstore = _stub
    warnings.warn("Stubbed missing megatron.core.dist_checkpointing.strategies.tensorstore")

# Monkey-patch TransformerConfig.__post_init__ to handle the legacy enable_cuda_graph field
# before importing anything else that triggers the dataclass initialization
try:
    from megatron.core.transformer.transformer_config import TransformerConfig
    _original_post_init = TransformerConfig.__post_init__

    def _patched_post_init(self):
        # If enable_cuda_graph is set (legacy 25.09 config) and cuda_graph_impl is also set
        # to something other than "none", resolve the conflict by disabling enable_cuda_graph
        # and letting cuda_graph_impl take precedence
        if getattr(self, 'enable_cuda_graph', False) and getattr(self, 'cuda_graph_impl', 'none') != 'none':
            warnings.warn(
                f'Resolving cuda_graph conflict: enable_cuda_graph=True with '
                f'cuda_graph_impl={self.cuda_graph_impl}. Clearing enable_cuda_graph.'
            )
            self.enable_cuda_graph = False
        _original_post_init(self)

    TransformerConfig.__post_init__ = _patched_post_init
except ImportError:
    pass

# Monkey-patch get_megatron_optimizer to strip kwargs not in its signature.
# NeMo 25.09 passes no_weight_decay_cond but some Megatron-Core versions don't accept it.
try:
    import megatron.core.optimizer as _mcore_optim
    _orig_get_megatron_optimizer = _mcore_optim.get_megatron_optimizer
    _sig = inspect.signature(_orig_get_megatron_optimizer)
    _has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in _sig.parameters.values()
    )
    if not _has_var_keyword:
        _accepted_params = set(_sig.parameters.keys())

        def _patched_get_megatron_optimizer(*args, **kwargs):
            filtered = {k: v for k, v in kwargs.items() if k in _accepted_params}
            stripped = set(kwargs) - set(filtered)
            if stripped:
                warnings.warn(
                    f"Stripped unsupported kwargs from get_megatron_optimizer: {stripped}"
                )
            return _orig_get_megatron_optimizer(*args, **filtered)

        _mcore_optim.get_megatron_optimizer = _patched_get_megatron_optimizer
except (ImportError, AttributeError):
    pass

# Now run the fdl_runner as normal
from nemo_run.core.runners.fdl_runner import fdl_runner_app
fdl_runner_app()
