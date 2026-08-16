import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
functional = torch.nn.functional

from mixing_matters.sink_block import block_attention_sink  # noqa: E402


class NemotronHAttention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        fail_after_attention: bool = False,
    ) -> torch.Tensor:
        output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=is_causal,
        )
        if fail_after_attention:
            raise ValueError("forward failed")
        return output


class LlamaAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_mask: torch.Tensor | None = None

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.last_mask = attention_mask
        return hidden_states


class ReluBlock(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.relu()


def _attention_inputs(
    query_length: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.zeros(1, 1, query_length, 1)
    key = torch.zeros(1, 1, 3, 1)
    value = torch.tensor([[[[1.0], [2.0], [4.0]]]])
    return query, key, value


def test_nemotron_block_changes_output_and_preserves_first_causal_row() -> None:
    module = NemotronHAttention()
    query, key, value = _attention_inputs()
    baseline = module(query, key, value, is_causal=True)

    with block_attention_sink(module) as info:
        blocked = module(query, key, value, is_causal=True)

    assert torch.equal(blocked[..., 0, :], baseline[..., 0, :])
    assert torch.isfinite(blocked).all()
    assert not torch.equal(blocked[..., 1:, :], baseline[..., 1:, :])
    assert info == {
        "attention_modules_hooked": 1,
        "nemotron_modules_wrapped": 1,
        "nemotron_modules_called": 1,
        "sdpa_calls": 1,
        "modified_query_rows": 2,
    }


def test_nemotron_block_composes_existing_additive_causal_mask() -> None:
    module = NemotronHAttention()
    query, key, value = _attention_inputs()
    minimum = torch.finfo(query.dtype).min
    attention_mask = torch.full((1, 1, 3, 3), minimum)
    attention_mask[..., 0, 0] = 0
    attention_mask[..., 1, :2] = 0
    attention_mask[..., 2, :] = 0

    with block_attention_sink(module) as info:
        blocked = module(query, key, value, attention_mask=attention_mask)

    assert torch.isfinite(blocked).all()
    assert blocked[..., 0, :].item() == pytest.approx(1.0)
    assert blocked[..., 1, :].item() == pytest.approx(2.0)
    assert info["modified_query_rows"] == 2


def test_nemotron_block_materializes_existing_mask_with_causality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = NemotronHAttention()
    query, key, value = _attention_inputs()
    attention_mask = torch.ones(3, 3, dtype=torch.bool)
    original_sdpa = functional.scaled_dot_product_attention
    observed: dict[str, torch.Tensor | bool] = {}

    def observe_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        observed["attention_mask"] = attn_mask.clone()
        observed["is_causal"] = is_causal
        return original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    monkeypatch.setattr(functional, "scaled_dot_product_attention", observe_sdpa)
    with block_attention_sink(module):
        blocked = module(query, key, value, attention_mask=attention_mask, is_causal=True)

    expected_mask = torch.tensor([[True, False, False], [False, True, False], [False, True, True]])
    assert observed["is_causal"] is False
    assert torch.equal(observed["attention_mask"], expected_mask)
    assert torch.allclose(blocked.flatten(), torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isfinite(blocked).all()


def test_nemotron_block_handles_cached_single_query() -> None:
    module = NemotronHAttention()
    query, key, value = _attention_inputs(query_length=1)
    baseline = module(query, key, value)

    with block_attention_sink(module) as info:
        blocked = module(query, key, value)

    assert blocked.item() == pytest.approx(3.0)
    assert not torch.equal(blocked, baseline)
    assert torch.isfinite(blocked).all()
    assert info["modified_query_rows"] == 1


def test_nemotron_block_handles_cached_causal_single_query() -> None:
    module = NemotronHAttention()
    query, key, value = _attention_inputs(query_length=1)

    with block_attention_sink(module) as info:
        blocked = module(query, key, value, is_causal=True)

    assert blocked.item() == pytest.approx(3.0)
    assert torch.isfinite(blocked).all()
    assert info["modified_query_rows"] == 1


def test_nemotron_block_rejects_shorter_key_sequence() -> None:
    module = NemotronHAttention()
    query = torch.zeros(1, 1, 3, 1)
    key = torch.zeros(1, 1, 2, 1)
    value = torch.ones(1, 1, 2, 1)

    with pytest.raises(RuntimeError, match="no SDPA call"):
        with block_attention_sink(module):
            with pytest.raises(ValueError, match="key length"):
                module(query, key, value)


def test_nemotron_block_requires_every_module_to_run() -> None:
    model = nn.ModuleList([NemotronHAttention(), NemotronHAttention()])
    query, key, value = _attention_inputs()

    with pytest.raises(RuntimeError, match="1 Nemotron-H modules"):
        with block_attention_sink(model):
            model[0](query, key, value)


def test_nemotron_block_requires_each_module_to_modify_a_row() -> None:
    model = nn.ModuleList([NemotronHAttention(), NemotronHAttention()])
    single_query = torch.zeros(1, 1, 1, 1)
    single_key = torch.zeros(1, 1, 1, 1)
    single_value = torch.ones(1, 1, 1, 1)
    query, key, value = _attention_inputs()

    with pytest.raises(RuntimeError, match="modified no query rows in 1"):
        with block_attention_sink(model) as info:
            output = model[0](single_query, single_key, single_value)
            assert torch.isfinite(output).all()
            model[1](query, key, value, is_causal=True)
    assert info["sdpa_calls"] == 2
    assert info["modified_query_rows"] == 2


def test_nemotron_forward_and_sdpa_restore_after_exception() -> None:
    module = NemotronHAttention()
    query, key, value = _attention_inputs()
    original_forward_function = module.forward.__func__
    original_sdpa = functional.scaled_dot_product_attention

    non_finite_value = value.clone()
    non_finite_value[..., 0, :] = torch.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        with block_attention_sink(module):
            module(query, key, non_finite_value)
            assert functional.scaled_dot_product_attention is original_sdpa
    assert functional.scaled_dot_product_attention is original_sdpa

    with block_attention_sink(module):
        with pytest.raises(ValueError, match="forward failed"):
            module(query, key, value, fail_after_attention=True)
        assert functional.scaled_dot_product_attention is original_sdpa

    assert "forward" not in module.__dict__
    assert module.forward.__func__ is original_forward_function
    assert functional.scaled_dot_product_attention is original_sdpa


def test_llama_mask_prehook_behavior_is_preserved() -> None:
    module = LlamaAttention()
    hidden_states = torch.randn(1, 3, 4)
    attention_mask = torch.zeros(1, 1, 3, 3)

    with block_attention_sink(module):
        module(hidden_states, attention_mask=attention_mask)

    assert module.last_mask is not attention_mask
    assert (module.last_mask[..., 0] < -1e10).all().item()
    module(hidden_states, attention_mask=attention_mask)
    assert torch.equal(module.last_mask, attention_mask)


def test_block_raises_when_no_attention_modules_present() -> None:
    model = nn.Sequential(ReluBlock(), ReluBlock())
    with pytest.raises(RuntimeError, match="no dense-attention modules"):
        with block_attention_sink(model):
            pass
