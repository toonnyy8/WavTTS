import torch
from torch import nn


def test_import():
    from wavtts.model import CFM, DiT  # noqa: F401

    assert torch.tensor(1.0).item() == 1.0


def _reinit_nonzero(model):
    # zero-init 的 AdaLN/proj_out 會讓輸出恆為 0，正負分支無差異；測試前擾動權重
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)


def test_dit_forward_shape():
    from wavtts.model.backbones.dit import STATE_CLEAN, DiT

    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    x = torch.randn(2, 1600)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)
    out = dit(x=x, state=state, time=torch.tensor(0.5))
    assert out.shape == (2, 1600)
    assert torch.isfinite(out).all()


def test_dit_cfg_infer_packs_pos_neg():
    from wavtts.model.backbones.dit import STATE_CLEAN, STATE_MIXED, DiT

    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    _reinit_nonzero(dit)
    x = torch.randn(2, 1600)
    state = torch.full((2,), STATE_CLEAN, dtype=torch.long)
    neg = torch.full((2,), STATE_MIXED, dtype=torch.long)
    out = dit(x=x, state=state, time=torch.tensor(0.5), cfg_infer=True, neg_state=neg)
    assert out.shape == (4, 1600)
    pos, negp = torch.chunk(out, 2, dim=0)
    assert not torch.allclose(pos, negp)  # 不同 state 必須產生不同輸出


def test_dit_state_changes_output():
    from wavtts.model.backbones.dit import STATE_CLEAN, STATE_MIXED, DiT

    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    _reinit_nonzero(dit)
    x = torch.randn(1, 1600)
    t = torch.tensor(0.5)
    out_clean = dit(x=x, state=torch.tensor([STATE_CLEAN]), time=t)
    out_mixed = dit(x=x, state=torch.tensor([STATE_MIXED]), time=t)
    assert not torch.allclose(out_clean, out_mixed)
