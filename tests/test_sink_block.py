import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mixing_matters.sink_block import block_attention_sink


class NemotronHAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_mask = None

    def forward(self, hidden_states, attention_mask=None):
        self.last_mask = attention_mask
        return hidden_states


class ReluBlock(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return hidden_states.relu()


class HybridModel(nn.Module):
    def __init__(self, n_attention: int = 3):
        super().__init__()
        self.attentions = nn.ModuleList([NemotronHAttention() for _ in range(n_attention)])
        self.other = nn.ModuleList([ReluBlock() for _ in range(2)])

    def forward(self, hidden_states, attention_mask=None):
        for module in self.attentions:
            hidden_states = module(hidden_states, attention_mask=attention_mask)
        for module in self.other:
            hidden_states = module(hidden_states, attention_mask=attention_mask)
        return hidden_states


def test_block_hooks_every_attention_module_and_zeros_token_zero_column():
    model = HybridModel(n_attention=3)
    hidden_states = torch.randn(1, 5, 4)
    attention_mask = torch.zeros(1, 1, 5, 5)
    with block_attention_sink(model) as info:
        assert info["attention_modules_hooked"] == 3
        _ = model(hidden_states, attention_mask=attention_mask)
    for module in model.attentions:
        # Token-0 column of the last-seen mask must be replaced with neg_inf.
        assert module.last_mask is not attention_mask
        assert torch.isfinite(module.last_mask).all().item() is False or (module.last_mask[..., 0] < -1e10).all().item()


def test_block_restores_forward_on_exit():
    model = HybridModel(n_attention=2)
    hidden_states = torch.randn(1, 4, 4)
    attention_mask = torch.zeros(1, 1, 4, 4)
    with block_attention_sink(model):
        pass
    _ = model(hidden_states, attention_mask=attention_mask)
    for module in model.attentions:
        # After the context exits, the model receives the untouched mask.
        assert torch.equal(module.last_mask, attention_mask)


def test_block_raises_when_no_attention_modules_present():
    model = nn.Sequential(ReluBlock(), ReluBlock())
    with pytest.raises(RuntimeError):
        with block_attention_sink(model):
            pass


def test_block_handles_positional_attention_mask():
    class PositionalModule(nn.Module):
        pass

    class LlamaAttention(nn.Module):  # noqa: N801 - matches transformers class name
        def __init__(self):
            super().__init__()
            self.last_mask = None

        def forward(self, hidden_states, attention_mask):
            self.last_mask = attention_mask
            return hidden_states

    class WithLlama(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = LlamaAttention()

    model = WithLlama()
    hidden_states = torch.randn(1, 3, 4)
    attention_mask = torch.zeros(1, 1, 3, 3)
    with block_attention_sink(model):
        _ = model.a(hidden_states, attention_mask)
    assert (model.a.last_mask[..., 0] < -1e10).all().item()
