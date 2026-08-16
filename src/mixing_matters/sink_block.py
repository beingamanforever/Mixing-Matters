"""Attention-sink blocking for dense-attention model families.

Llama and Qwen attention modules accept an additive attention mask directly,
so a forward-pre hook can block token 0. The pinned Nemotron-H attention
implementation may build its causal behavior only inside SDPA. Its bound
forward is therefore wrapped so the mask is changed at the actual score call.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

_MASK_HOOK_MODULE_NAMES = (
    "LlamaAttention",
    "LlamaSdpaAttention",
    "LlamaFlashAttention2",
    "Qwen2Attention",
    "Qwen2SdpaAttention",
    "Qwen2FlashAttention2",
)
_NEMOTRON_ATTENTION_MODULE_NAME = "NemotronHAttention"
_TARGET_ATTENTION_MODULE_NAMES = (
    *_MASK_HOOK_MODULE_NAMES,
    _NEMOTRON_ATTENTION_MODULE_NAME,
)


@dataclass
class _NemotronModuleState:
    """Hold asynchronous validation tensors for one Nemotron attention module."""

    calls: int = 0
    modified_rows: Any = None
    outputs_finite: Any = None


def _module_is_attention(module: Any) -> bool:
    """Return whether ``module`` is a supported dense-attention block."""
    return type(module).__name__ in _TARGET_ATTENTION_MODULE_NAMES


def _iter_attention_modules(model: Any) -> Iterable[Any]:
    """Yield supported attention modules from ``model``."""
    for module in model.modules():
        if _module_is_attention(module):
            yield module


def _rewrite_attention_mask(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return forward arguments with the attention-mask token-0 column blocked."""
    import torch

    def mask_tensor(tensor: Any) -> Any:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return tensor
        blocked = tensor.clone()
        if blocked.dtype == torch.bool:
            blocked[..., 0] = False
            return blocked
        minimum = torch.finfo(blocked.dtype).min if blocked.is_floating_point() else -1e9
        blocked[..., 0] = minimum
        return blocked

    new_kwargs = dict(kwargs)
    if "attention_mask" in new_kwargs:
        new_kwargs["attention_mask"] = mask_tensor(new_kwargs["attention_mask"])
        return args, new_kwargs
    if len(args) < 2:
        return args, kwargs
    new_args = list(args)
    new_args[1] = mask_tensor(args[1])
    return tuple(new_args), kwargs


def _causal_legal_mask(query: Any, key: Any) -> Any:
    """Return a bottom-right causal layout for current and cached queries."""
    import torch

    query_length = query.shape[-2]
    key_length = key.shape[-2]
    if key_length < query_length:
        raise ValueError("sink-block requires key length to be at least the query length")
    query_positions = torch.arange(query_length, device=query.device)
    key_positions = torch.arange(key_length, device=query.device)
    query_start = key_length - query_length
    return key_positions.unsqueeze(0) <= (query_positions + query_start).unsqueeze(1)


def _mask_legal_entries(attention_mask: Any) -> Any:
    """Return which entries of a boolean or additive SDPA mask are legal."""
    import torch

    if attention_mask.dtype == torch.bool:
        return attention_mask
    if attention_mask.is_floating_point():
        minimum = torch.finfo(attention_mask.dtype).min
        return torch.isfinite(attention_mask) & (attention_mask > minimum / 2)
    return attention_mask != 0


def _block_sdpa_sink(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
    """Compose an SDPA mask with token-0 blocking where another key is legal."""
    import torch

    query, key = args[:2]
    if key.shape[-2] < query.shape[-2]:
        raise ValueError("sink-block requires key length to be at least the query length")
    attention_mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
    is_causal = bool(kwargs.get("is_causal", args[5] if len(args) > 5 else False))

    if attention_mask is None:
        legal_entries = torch.ones(
            query.shape[-2], key.shape[-2], dtype=torch.bool, device=query.device
        )
        blocked_mask = legal_entries.clone()
    else:
        legal_entries = _mask_legal_entries(attention_mask)
        blocked_mask = attention_mask.clone()
    if is_causal:
        legal_entries = legal_entries & _causal_legal_mask(query, key)
        if blocked_mask.dtype == torch.bool:
            blocked_mask = legal_entries.clone()
        else:
            minimum = torch.finfo(blocked_mask.dtype).min
            blocked_mask = torch.where(legal_entries, blocked_mask, minimum)

    key_zero_is_legal = legal_entries[..., 0]
    has_alternative_key = legal_entries[..., 1:].any(dim=-1)
    modified_rows = key_zero_is_legal & has_alternative_key
    if blocked_mask.dtype == torch.bool:
        blocked_mask[..., 0] &= ~modified_rows
    else:
        minimum = torch.finfo(blocked_mask.dtype).min
        blocked_mask[..., 0] = torch.where(modified_rows, minimum, blocked_mask[..., 0])

    new_args = list(args)
    new_kwargs = dict(kwargs)
    if "attn_mask" in new_kwargs:
        new_kwargs["attn_mask"] = blocked_mask
    elif len(new_args) > 3:
        new_args[3] = blocked_mask
    else:
        new_kwargs["attn_mask"] = blocked_mask
    if is_causal:
        if "is_causal" in new_kwargs:
            new_kwargs["is_causal"] = False
        elif len(new_args) > 5:
            new_args[5] = False
    return tuple(new_args), new_kwargs, modified_rows.sum()


def _wrap_nemotron_forward(
    module: Any, info: dict[str, int], module_states: dict[int, _NemotronModuleState]
) -> Any:
    """Install a bound-forward wrapper that intercepts only this module's SDPA."""
    import torch
    import torch.nn.functional as functional

    original_forward = module.forward

    @wraps(original_forward)
    def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        original_sdpa = functional.scaled_dot_product_attention

        def blocked_sdpa(*sdpa_args: Any, **sdpa_kwargs: Any) -> Any:
            blocked_args, blocked_kwargs, modified_rows = _block_sdpa_sink(sdpa_args, sdpa_kwargs)
            module_id = id(module)
            state = module_states[module_id]
            if state.calls == 0:
                info["nemotron_modules_called"] += 1
            state.calls += 1
            info["sdpa_calls"] += 1
            result = original_sdpa(*blocked_args, **blocked_kwargs)
            outputs_finite = torch.isfinite(result).all()
            if state.modified_rows is None:
                state.modified_rows = modified_rows
                state.outputs_finite = outputs_finite
            else:
                state.modified_rows = state.modified_rows + modified_rows
                state.outputs_finite = state.outputs_finite & outputs_finite
            return result

        functional.scaled_dot_product_attention = blocked_sdpa
        try:
            return original_forward(*args, **kwargs)
        finally:
            functional.scaled_dot_product_attention = original_sdpa

    module.forward = wrapped_forward
    return original_forward


def _validate_nemotron_states(
    states: dict[int, _NemotronModuleState], info: dict[str, int]
) -> None:
    """Materialize accumulated device telemetry and reject invalid interventions."""
    import torch

    missing_calls = sum(state.calls == 0 for state in states.values())
    if missing_calls:
        raise RuntimeError(
            f"sink-block observed no SDPA call in {missing_calls} Nemotron-H modules"
        )
    if not states:
        return

    device_summary = torch.stack(
        [
            torch.stack((state.modified_rows, state.outputs_finite.to(state.modified_rows.dtype)))
            for state in states.values()
        ]
    )
    summary = device_summary.detach().cpu().tolist()
    modified_rows = [int(row[0]) for row in summary]
    info["modified_query_rows"] = sum(modified_rows)
    non_finite_modules = sum(not bool(row[1]) for row in summary)
    if non_finite_modules:
        raise RuntimeError(
            f"sink-block SDPA produced non-finite output in {non_finite_modules} Nemotron-H modules"
        )
    unmodified_modules = sum(row_count == 0 for row_count in modified_rows)
    if unmodified_modules:
        raise RuntimeError(
            f"sink-block modified no query rows in {unmodified_modules} Nemotron-H modules"
        )


@contextmanager
def block_attention_sink(model: Any) -> Iterator[dict[str, int]]:
    """Temporarily hide token 0 from every supported attention block.

    Nemotron-H interception reports the number of SDPA calls and query rows
    changed. A query row that has no other legal key retains token 0 so its
    attention output remains finite. All hooks and bound forwards are restored
    on normal exit and exceptions. The Nemotron wrapper temporarily replaces
    the process-global SDPA function during each bound forward, so this context
    supports synchronous inference only.

    Raises:
        RuntimeError: If the model has no supported dense-attention modules.
    """
    handles = []
    wrapped_modules: list[tuple[Any, Any, bool]] = []
    modules = list(_iter_attention_modules(model))
    if not modules:
        raise RuntimeError(
            "sink-block found no dense-attention modules to hook; the model "
            "family does not carry attention layers and cannot be ablated"
        )

    info = {
        "attention_modules_hooked": len(modules),
        "nemotron_modules_wrapped": 0,
        "nemotron_modules_called": 0,
        "sdpa_calls": 0,
        "modified_query_rows": 0,
    }
    module_states: dict[int, _NemotronModuleState] = {}
    for module in modules:
        if type(module).__name__ == _NEMOTRON_ATTENTION_MODULE_NAME:
            had_instance_forward = "forward" in module.__dict__
            module_states[id(module)] = _NemotronModuleState()
            original_forward = _wrap_nemotron_forward(module, info, module_states)
            wrapped_modules.append((module, original_forward, had_instance_forward))
            info["nemotron_modules_wrapped"] += 1
            continue

        def hook(
            _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return _rewrite_attention_mask(args, kwargs)

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    try:
        yield info
    finally:
        for handle in handles:
            handle.remove()
        for module, original_forward, had_instance_forward in wrapped_modules:
            if had_instance_forward:
                module.forward = original_forward
            else:
                del module.__dict__["forward"]
    _validate_nemotron_states(module_states, info)
