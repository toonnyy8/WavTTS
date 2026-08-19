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


def make_model(use_aux_mel_loss=False, **kwargs):
    from wavtts.model import CFM, DiT

    torch.manual_seed(0)
    transformer = DiT(dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, wav_frame_len=160)
    defaults = dict(
        waveform_kwargs={"wav_frame_len": 160},
        prediction="x_pred",
        loss_space="v",
        t_eps=0.02,
        use_aux_mel_loss=use_aux_mel_loss,
        aux_mel_loss_weight=0.05,
        sample_rate=16000,
        latents_scale=9.0,
    )
    defaults.update(kwargs)
    return CFM(transformer=transformer, **defaults)


def test_mix_augment_labels_and_content():
    from wavtts.model.backbones.dit import STATE_CLEAN, STATE_MIXED

    model = make_model(p_mix=1.0)
    torch.manual_seed(0)
    x = torch.randn(4, 3200)
    lens = torch.full((4,), 3200, dtype=torch.long)
    x_aug, state = model._mix_augment(x, lens)
    assert (state == STATE_MIXED).all()
    assert not torch.allclose(x_aug, x)

    model.p_mix = 0.0
    x_aug, state = model._mix_augment(x, lens)
    assert (state == STATE_CLEAN).all()
    assert torch.equal(x_aug, x)


def test_mix_augment_concat_prefix_preserved():
    model = make_model(p_mix=1.0, p_concat=1.0)
    torch.manual_seed(0)
    x = torch.randn(2, 3200)
    lens = torch.full((2,), 3200, dtype=torch.long)
    x_aug, _state = model._mix_augment(x, lens)
    # 切換點最早在 0.3*3200=960，之前的內容必須原封不動（no leaky：前段就是原語者）
    assert torch.equal(x_aug[:, :900], x[:, :900])


def test_mix_augment_batch_of_one_is_noop():
    from wavtts.model.backbones.dit import STATE_CLEAN

    model = make_model(p_mix=1.0)
    x = torch.randn(1, 3200)
    lens = torch.full((1,), 3200, dtype=torch.long)
    x_aug, state = model._mix_augment(x, lens)
    assert torch.equal(x_aug, x)
    assert (state == STATE_CLEAN).all()


def test_train_step_backward():
    model = make_model()
    torch.manual_seed(0)
    wav = torch.randn(4, 16000) * 0.1
    lens = torch.tensor([16000, 12000, 16000, 8000])
    loss, loss_dict = model(wav, lens=lens)
    assert torch.isfinite(loss)
    assert set(loss_dict) == {"total_loss", "flow_loss", "aux_mel_loss"}
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_train_step_with_aux_mel_loss():
    model = make_model(use_aux_mel_loss=True)
    torch.manual_seed(0)
    wav = torch.randn(2, 16000) * 0.1
    loss, loss_dict = model(wav)
    assert torch.isfinite(loss)
    assert loss_dict["aux_mel_loss"].item() != 0.0
    loss.backward()
