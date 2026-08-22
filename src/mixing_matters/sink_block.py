"""Attention-sink blocking for the Phase 7 sink-ablation experiment.

The sink hypothesis says that dense-attention causal transformers place
disproportionate attention mass on the first token of the prompt and
that mass drives the primacy arm on the position curve. This module
supplies a context manager that installs forward-pre hooks on the
attention modules of a hybrid or dense-attention model and rewrites
their attention masks so token 0 is treated as if it were fully masked.
The masked position is picked before the softmax so the remaining
positions renormalize; nothing is scaled by a hand-picked constant.

The hooks target only ``torch.nn.Module`` classes that expose an
``attention_mask`` argument on their forward call, which covers the
Llama, Qwen, and Nemotron-H hybrid attention blocks in transformers
4.57.x. Pure-SSM families (``mamba``, ``mamba2``) have no attention
call sites to intercept and are rejected up front.

The context manager restores every hook on exit, including on
exceptions.
"""

from collections.abc import Iterable
from contextlib import contextmanager

_TARGET_ATTENTION_MODULE_NAMES = (
    # Llama-3.1 and Qwen2.5 attention module classes.
    "LlamaAttention",
    "LlamaSdpaAttention",
    "LlamaFlashAttention2",
    "Qwen2Attention",
    "Qwen2SdpaAttention",
    "Qwen2FlashAttention2",
    # Nemotron-H hybrid attention module class (custom code, class name is
    # stable across the pinned revision).
    "NemotronHAttention",
)


def _module_is_attention(module) -> bool:
    """Return True when ``module`` is a dense-attention block class."""
    return type(module).__name__ in _TARGET_ATTENTION_MODULE_NAMES


def _iter_attention_modules(model) -> Iterable:
    for module in model.modules():
        if _module_is_attention(module):
            yield module


def _rewrite_attention_mask(args, kwargs, mask_first_position: bool = True):
    """Return ``(args, kwargs)`` with the attention_mask token-0 column zeroed.

    Attention masks in transformers are usually ``(batch, 1, q_len, k_len)``
    additive masks where 0.0 keeps a position and a large negative value
    masks it. The token-0 column of ``k_len`` is set to the same large
    negative value so every query position ignores it.
    """
    import torch

    def _mask_tensor(tensor):
        if tensor is None:
            return tensor
        if not isinstance(tensor, torch.Tensor):
            return tensor
        neg_inf = torch.finfo(tensor.dtype).min if tensor.is_floating_point() else -1e9
        clone = tensor.clone()
        # Support (batch, q) boolean masks and 4D additive masks alike.
        if clone.dim() == 2:
            if clone.dtype == torch.bool:
                clone[:, 0] = False
            else:
                clone[:, 0] = neg_inf
        elif clone.dim() == 4:
            if clone.dtype == torch.bool:
                clone[..., 0] = False
            else:
                clone[..., 0] = neg_inf
        else:
            clone[..., 0] = neg_inf
        return clone

    new_kwargs = dict(kwargs)
    if "attention_mask" in new_kwargs:
        new_kwargs["attention_mask"] = _mask_tensor(new_kwargs["attention_mask"])
        return args, new_kwargs
    # Positional attention_mask argument: forward signatures put it at index 1.
    if len(args) >= 2:
        new_args = list(args)
        new_args[1] = _mask_tensor(args[1])
        return tuple(new_args), kwargs
    return args, kwargs


@contextmanager
def block_attention_sink(model):
    """Context manager that hides token 0 from every attention block.

    On enter, walks ``model.modules()`` and registers a forward-pre hook on
    every attention module. Each hook rewrites the ``attention_mask``
    argument so token 0 becomes fully masked. On exit, every hook is
    removed and the model returns to its original forward behaviour.

    Raises ``RuntimeError`` when no attention modules are found so a
    pure-SSM model does not silently no-op through the ablation.
    """
    handles = []
    matched = 0
    for module in _iter_attention_modules(model):
        matched += 1

        def hook(_module, args, kwargs):
            return _rewrite_attention_mask(args, kwargs)

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    if matched == 0:
        raise RuntimeError(
            "sink-block found no dense-attention modules to hook; the model "
            "family does not carry attention layers and cannot be ablated"
        )
    try:
        yield {"attention_modules_hooked": matched}
    finally:
        for handle in handles:
            handle.remove()
