# WavTTS 無條件語音生成 + 混合語音負樣本 CFG — 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 WavTTS 就地改造為無條件純語音生成模型：移除 text/audio-cond，唯一條件是 clean/mixed/null 三值狀態標記；訓練時以 no-leaky 混合增強（overlap + concat 兩型態）產生 mixed 負樣本；推論時以 `v = v_clean + w·(v_clean − v_mixed)` 做負樣本 CFG。

**Architecture:** DiT backbone 的 conditioning 改為 timestep embedding 加上 `nn.Embedding(3, dim)` 狀態標記；CFM 移除 infilling，loss 覆蓋整段有效長度，`_mix_augment` 在 batch 內 roll 出 partner 做等功率 overlap 混合或等功率 crossfade concat 串接。就地修改，TTS 的 `infer_cli.py`/`utils_infer.py`/`eval/`/`mmdit.py` 保留原檔但不再被新程式 import、不維護。

**Tech Stack:** PyTorch、torchdiffeq (Euler ODE)、Hydra/OmegaConf、accelerate、pytest。

**Spec:** `docs/superpowers/specs/2026-08-19-uncond-speech-cfg-design.md`

## Global Constraints

- Python 3.10；不新增任何依賴（pyproject 現有依賴即全部所需，pytest 在 `[dev]` extra）。
- 測試一律在專案 venv 內執行：`.venv/bin/python -m pytest ...`（Task 1 建立）。CPU 即可，測試模型一律縮小（dim=64, depth=2）。
- `wav_frame_len: 160`、`sample_rate: 16000` 不變。
- 狀態常數固定為 `STATE_CLEAN = 0`、`STATE_MIXED = 1`、`STATE_NULL = 2`、`NUM_STATES = 3`，定義於 `src/wavtts/model/backbones/dit.py`，全 repo 由此 import。
- CFM 混合增強預設值（與 spec 一致）：`p_mix=0.5`、`p_concat=0.5`、`mix_lambda_range=(0.3, 0.7)`、`concat_point_range=(0.3, 0.7)`、`concat_xfade_ms=20`、`state_drop_prob=0.1`。
- 不改動：`src/wavtts/infer/infer_cli.py`、`src/wavtts/infer/utils_infer.py`、`src/wavtts/eval/`、`src/wavtts/model/backbones/mmdit.py`、`src/wavtts/model/modules.py`、`src/wavtts/model/utils.py`、`src/wavtts/train/datasets/`。
- 所有測試集中在 `tests/test_uncond_smoke.py` 一個檔案，逐 task 追加。
- Commit message 結尾附 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。

---

### Task 1: 測試環境

**Files:**
- Create: `.venv/`（uv venv，不進 git）
- Create: `tests/test_uncond_smoke.py`（先放一個 import 冒煙測試）

**Interfaces:**
- Produces: 可用的 `.venv/bin/python -m pytest`；後續所有 task 用它跑測試。

- [ ] **Step 1: 建 venv 並安裝**

```bash
cd /media/8tsp/projects/WavTTS
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -e ".[dev]"
```

注意：會下載 CPU 版 torch（數百 MB），耐心等。若 `torchcodec` 安裝失敗，改跑 `uv pip install --python .venv/bin/python -e ".[dev]" --no-deps` 後手動裝其餘依賴不可取——正確做法是單獨排除：`torchcodec` 只有 dataset 音訊解碼會用到，測試不會 import 它，可 `uv pip install --python .venv/bin/python torch torchaudio torchdiffeq x-transformers einops hydra-core omegaconf datasets accelerate ema_pytorch wandb pypinyin rjieba tqdm soundfile pytest && uv pip install --python .venv/bin/python -e . --no-deps` 作為 fallback。

- [ ] **Step 2: 寫 import 冒煙測試**

建立 `tests/test_uncond_smoke.py`：

```python
import torch


def test_import():
    from wavtts.model import CFM, DiT  # noqa: F401

    assert torch.tensor(1.0).item() == 1.0
```

- [ ] **Step 3: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py -v`
Expected: PASS（此時 import 的還是舊版 CFM/DiT，本 task 只驗環境）。

- [ ] **Step 4: Commit**

```bash
git add tests/test_uncond_smoke.py
git commit -m "test: add smoke test scaffold and dev venv setup

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: DiT backbone 就地改寫

**Files:**
- Modify: `src/wavtts/model/backbones/dit.py`（整檔改寫，內容如下）
- Test: `tests/test_uncond_smoke.py`

**Interfaces:**
- Produces:
  - 常數 `STATE_CLEAN = 0`、`STATE_MIXED = 1`、`STATE_NULL = 2`、`NUM_STATES = 3`。
  - `DiT.__init__(*, dim, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4, wav_frame_len=160, qk_norm=None, pe_attn_head=None, attn_backend="torch", attn_mask_enabled=False, long_skip_connection=False, checkpoint_activations=False, use_audio_proj=False, audio_proj_dim=None, audio_proj_hidden=None)`——**不再有** text_num_embeds/text_dim/text_mask_padding/conv_layers 等參數。
  - `DiT.forward(x, state, time, mask=None, cfg_infer=False, neg_state=None, lens=None) -> Tensor`：`x` 為 `[b, nw]` 波形、`state` 為 `[b]` long、回傳 `[b, nw]`；`cfg_infer=True` 時打包正/負分支回傳 `[2b, nw]`（前半正分支、後半負分支）。回傳單一 tensor（不再是 3-tuple）。
  - 保留 `set_wav_frame_len`、`_wav_to_tokens`、`_tokens_to_wav`。**移除** `clear_cache`、`TextEmbedding`、text cache。

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_uncond_smoke.py` 追加：

```python
from torch import nn


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
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py -v`
Expected: 新增三個測試 FAIL（`cannot import name 'STATE_CLEAN'` 或 `unexpected keyword argument 'state'`）。

- [ ] **Step 3: 改寫 `src/wavtts/model/backbones/dit.py`**

整檔取代為：

```python
"""
ein notation:
b - batch
n - sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from wavtts.model.modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
)


# speech-state conditioning: the only condition this model has
STATE_CLEAN = 0  # single consistent speaker
STATE_MIXED = 1  # speaker-inconsistent (overlap or concat augmentation)
STATE_NULL = 2  # unconditional
NUM_STATES = 3


# waveform patch embedding


class InputEmbedding(nn.Module):
    def __init__(
        self,
        wav_frame_len,
        out_dim,
        use_audio_proj: bool = False,
        audio_proj_dim: int | None = None,
        audio_proj_hidden: int | None = None,
    ):
        super().__init__()

        self.use_audio_proj = use_audio_proj

        if not use_audio_proj:
            self.proj = nn.Linear(wav_frame_len, out_dim)
        else:
            audio_proj_dim = out_dim if audio_proj_dim is None else audio_proj_dim
            audio_proj_hidden = audio_proj_dim if audio_proj_hidden is None else audio_proj_hidden
            self.x_proj = nn.Sequential(
                nn.Linear(wav_frame_len, audio_proj_hidden, bias=False),
                nn.Linear(audio_proj_hidden, audio_proj_dim),
            )
            self.fuse = nn.Linear(audio_proj_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], audio_mask: bool["b n"] | None = None):
        if not self.use_audio_proj:
            h = self.proj(x)
        else:
            h = self.fuse(self.x_proj(x))

        h = self.conv_pos_embed(h, mask=audio_mask) + h
        return h


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        wav_frame_len=160,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_audio_proj: bool = False,
        audio_proj_dim: int | None = None,
        audio_proj_hidden: int | None = None,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.state_embed = nn.Embedding(NUM_STATES, dim)
        self.input_embed = InputEmbedding(
            wav_frame_len,
            dim,
            use_audio_proj=use_audio_proj,
            audio_proj_dim=audio_proj_dim,
            audio_proj_hidden=audio_proj_hidden,
        )

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out_dim = wav_frame_len
        self.proj_out = nn.Linear(dim, wav_frame_len)
        self.proj_out_output_layer = self.proj_out

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

        # raw waveform tokenization.
        self.wav_frame_len = wav_frame_len

    def initialize_weights(self):
        # State (class) embedding:
        nn.init.normal_(self.state_embed.weight, std=0.02)

        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out_output_layer.weight, 0)
        if self.proj_out_output_layer.bias is not None:
            nn.init.constant_(self.proj_out_output_layer.bias, 0)

    def set_wav_frame_len(self, wav_frame_len: int):
        self.wav_frame_len = int(wav_frame_len)

        if self.wav_frame_len != self.proj_out_dim:
            raise ValueError(
                f"wav_frame_len ({self.wav_frame_len}) must equal proj_out_dim ({self.proj_out_dim}) "
                "for reshape wav front-end."
            )

    def _wav_to_tokens(
        self,
        wav: torch.Tensor,
        mask: torch.Tensor | None = None,
        lens: torch.Tensor | None = None,
    ):
        assert wav.ndim == 2, f"Expected [B, N] wav input, got {tuple(wav.shape)}"

        bsz, num_samples = wav.shape
        frame_len = self.wav_frame_len
        pad_len = (frame_len - (num_samples % frame_len)) % frame_len

        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)

        tokens = wav.view(bsz, -1, frame_len)

        token_mask = None
        if mask is not None:
            token_mask = mask.view(bsz, -1, frame_len).any(dim=-1)

        token_lens = None
        if lens is not None:
            token_lens = (lens.to(dtype=torch.long, device=wav.device) + frame_len - 1) // frame_len

        return tokens, token_mask, token_lens

    def _tokens_to_wav(self, tokens: torch.Tensor, target_num_samples: int):
        wav = tokens.reshape(tokens.shape[0], -1)
        return wav[:, :target_num_samples]

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        x: float["b nw"],  # noised waveform
        state: int["b"],  # speech-state condition: STATE_CLEAN / STATE_MIXED / STATE_NULL
        time: float["b"] | float[""],  # time step
        mask: bool["b nw"] | None = None,
        cfg_infer: bool = False,  # pack positive & negative state forward
        neg_state: int["b"] | None = None,  # negative branch state for cfg_infer
        lens: int["b"] | None = None,
    ):
        if x.ndim != 2:
            raise ValueError(f"WavTTS DiT expects raw waveform x [B, N], got {x.ndim}D.")

        target_num_samples = x.shape[1]
        x, token_mask, _token_lens = self._wav_to_tokens(x, mask=mask, lens=lens)
        mask = token_mask

        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        h = self.input_embed(x, audio_mask=mask)

        if cfg_infer:  # pack positive & negative state forward: b n d -> 2b n d
            if neg_state is None:
                neg_state = torch.full_like(state, STATE_NULL)
            h = torch.cat((h, h), dim=0)
            t = torch.cat((t, t), dim=0)
            state = torch.cat((state, neg_state), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None

        t = t + self.state_embed(state)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = h

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                h = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), h, t, mask, rope, use_reentrant=False)
            else:
                h = block(h, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            h = self.long_skip_connection(torch.cat((h, residual), dim=-1))

        h = self.norm_out(h, t)
        output = self.proj_out(h)

        return self._tokens_to_wav(output, target_num_samples=target_num_samples)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py -v`
Expected: 本 task 三個新測試 PASS。`test_import` 此時會 FAIL——`wavtts/model/__init__.py` import 的 `CFM`（舊版）還在 import `mask_from_frac_lengths` 等，且舊 `cfm.py` 仍相容；實際上 `CFM` 舊版不依賴 dit 的 `TextEmbedding`，但 `CFM.sample` 會呼叫已刪除的 `transformer.clear_cache()`——import 階段不會爆。若 `test_import` 因其他原因 FAIL，記下原因，Task 3 改寫 `cfm.py` 後必須全綠；只要本 task 三個新測試 PASS 即可 commit。

- [ ] **Step 5: Commit**

```bash
git add src/wavtts/model/backbones/dit.py tests/test_uncond_smoke.py
git commit -m "refactor: strip text/cond from DiT, add clean/mixed/null state conditioning

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: CFM 混合增強與訓練 forward

**Files:**
- Modify: `src/wavtts/model/cfm.py`（整檔改寫，本 task 完成 `__init__`、`_mix_augment`、`forward`；`sample` 一併寫入但由 Task 4 測試）
- Test: `tests/test_uncond_smoke.py`

**Interfaces:**
- Consumes: Task 2 的 `DiT.forward(x, state, time, mask, lens)` 與 `STATE_*` 常數。
- Produces:
  - `CFM.__init__(transformer, sigma=0.0, odeint_kwargs=dict(method="euler"), p_mix=0.5, p_concat=0.5, mix_lambda_range=(0.3, 0.7), concat_point_range=(0.3, 0.7), concat_xfade_ms=20.0, state_drop_prob=0.1, waveform_kwargs=dict(), prediction="flow", loss_space="flow", t_sampling="uniform", P_mean=0.0, P_std=1.0, time_shift=1.0, t_eps=1e-4, use_aux_mel_loss=False, aux_mel_loss_weight=0.0, aux_mel_loss_masked=True, sample_rate=16000, latents_scale=1.0)`。
  - `CFM._mix_augment(x1, lens) -> (x1_aug, state)`：`x1` `[b, nw]`、`state` `[b]` long。
  - `CFM.forward(inp, *, lens=None) -> (total_loss, loss_dict)`——**2-tuple**，`loss_dict` 含 `total_loss`/`flow_loss`/`aux_mel_loss`。
  - `CFM.sample(duration, *, batch=1, steps=32, cfg_strength=2.0, negative="mixed", sway_sampling_coef=None, timestep_mapping="sway_sampling", timestep_power=None, shift=1.0, use_epss=True, seed=None) -> (out, trajectory)`，`out` 為 `[batch, duration]`。

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_uncond_smoke.py` 追加：

```python
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
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py -v`
Expected: 新測試 FAIL（舊 `CFM.__init__` 沒有 `p_mix` 參數）。

- [ ] **Step 3: 改寫 `src/wavtts/model/cfm.py`**

整檔取代為：

```python
"""
ein notation:
b - batch
n - sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint

from wavtts.model.backbones.dit import STATE_CLEAN, STATE_MIXED, STATE_NULL
from wavtts.model.modules import MelSpectrogramLoss
from wavtts.model.utils import exists, get_epss_timesteps, lens_to_mask


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            method="euler"  # 'midpoint'
        ),
        p_mix: float = 0.5,
        p_concat: float = 0.5,
        mix_lambda_range: tuple[float, float] = (0.3, 0.7),
        concat_point_range: tuple[float, float] = (0.3, 0.7),
        concat_xfade_ms: float = 20.0,
        state_drop_prob: float = 0.1,
        waveform_kwargs: dict = dict(),
        prediction: str = "flow",  # "flow" | "x_pred"
        loss_space: str = "flow",  # "flow" | "v" | "x"
        t_sampling: str = "uniform",  # "uniform" | "logistic_normal"
        P_mean: float = 0.0,
        P_std: float = 1.0,
        time_shift: float = 1.0,
        t_eps: float = 1e-4,
        use_aux_mel_loss: bool = False,
        aux_mel_loss_weight: float = 0.0,
        aux_mel_loss_masked: bool = True,
        sample_rate: int = 16000,
        latents_scale: float = 1.0,
    ):
        super().__init__()

        # waveform geometry
        waveform_kwargs = dict(waveform_kwargs)
        self.wav_frame_len = int(waveform_kwargs.pop("wav_frame_len", 160))
        self.num_channels = self.wav_frame_len

        # no-leaky mixing augmentation / state conditioning
        self.p_mix = p_mix
        self.p_concat = p_concat
        self.mix_lambda_range = tuple(mix_lambda_range)
        self.concat_point_range = tuple(concat_point_range)
        self.concat_xfade_ms = concat_xfade_ms
        self.state_drop_prob = state_drop_prob

        # transformer
        self.transformer = transformer
        self.dim = transformer.dim

        if hasattr(self.transformer, "set_wav_frame_len"):
            self.transformer.set_wav_frame_len(self.wav_frame_len)

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # enhanced flow model settings
        self.prediction = prediction
        self.loss_space = loss_space
        self.t_sampling = t_sampling
        self.P_mean = P_mean
        self.P_std = P_std
        self.time_shift = time_shift
        self.t_eps = t_eps
        self.latents_scale = latents_scale
        self.target_sample_rate = sample_rate
        # aux mel loss
        self.use_aux_mel_loss = use_aux_mel_loss
        self.aux_mel_loss_masked = aux_mel_loss_masked
        if self.use_aux_mel_loss:
            self.aux_mel_loss = MelSpectrogramLoss(
                sample_rate=sample_rate,
                n_mels=[5, 10, 20, 40, 80, 160, 320],
                window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
                mel_fmin=[0, 0, 0, 0, 0, 0, 0],
                mel_fmax=[None] * 7,
                pow=1.0,
                clamp_eps=1e-5,
                weight=aux_mel_loss_weight,
            )
        else:
            self.aux_mel_loss = None

    @property
    def device(self):
        return next(self.parameters()).device

    def _sample_time(self, batch: int, dtype, device):
        if self.t_sampling == "uniform":
            t = torch.rand((batch,), dtype=dtype, device=device)
        elif self.t_sampling == "logistic_normal":
            # JiT: t = sigmoid(N(P_mean, P_std))
            z = torch.randn((batch,), device=device, dtype=dtype) * self.P_std + self.P_mean
            t = torch.sigmoid(z)
        else:
            raise ValueError(f"Unknown t_sampling: {self.t_sampling}")

        if self.time_shift != 1.0:
            t = t / (t + self.time_shift * (1 - t))

        return t

    def _x_to_v(self, x_pred, z, t):
        # v_pred = (x_pred - z) / (1 - t)
        denom = (1.0 - t).clamp_min(self.t_eps)
        while denom.ndim < z.ndim:
            denom = denom.unsqueeze(-1)
        return (x_pred - z) / denom

    def _mix_augment(self, x1: float["b nw"], lens: int["b"]):
        """No-leaky mixing augmentation: content and state label always agree
        over the whole utterance, and boundaries carry no tell-tale artifacts.

        overlap: equal-power blend with a batch-roll partner (simultaneous speakers)
        concat:  equal-power crossfade into the partner at a random switch point
                 (temporal speaker switch)
        """
        batch, seq_len = x1.shape
        device = x1.device
        state = torch.full((batch,), STATE_CLEAN, device=device, dtype=torch.long)
        if batch < 2 or self.p_mix <= 0.0:
            return x1, state  # roll partner would be the sample itself

        partner = x1.roll(1, dims=0)
        mix_flags = torch.rand(batch, device=device) < self.p_mix
        concat_flags = torch.rand(batch, device=device) < self.p_concat

        # overlap: x = sqrt(1-lam)*x1 + sqrt(lam)*partner
        lo, hi = self.mix_lambda_range
        lam = torch.empty((batch, 1), device=device, dtype=x1.dtype).uniform_(lo, hi)
        overlap = torch.sqrt(1.0 - lam) * x1 + torch.sqrt(lam) * partner

        # concat: switch to partner at s with an equal-power crossfade
        # (a hard cut's click would let the model detect "mixed" from the boundary
        #  artifact instead of speaker identity, breaking the CFG direction)
        xfade_len = max(1, int(self.concat_xfade_ms * self.target_sample_rate / 1000.0))
        plo, phi = self.concat_point_range
        u = torch.empty((batch,), device=device, dtype=x1.dtype).uniform_(plo, phi)
        s = (u * lens.to(x1.dtype)).long().clamp(min=1, max=seq_len - 1)
        idx = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, n]
        prog = ((idx - s.unsqueeze(-1)).to(x1.dtype) / xfade_len).clamp(0.0, 1.0)
        g_in = torch.sin(prog * math.pi / 2)  # partner fades in
        g_out = torch.cos(prog * math.pi / 2)  # original fades out; g_in^2 + g_out^2 = 1
        concat = g_out * x1 + g_in * partner

        mixed = torch.where(concat_flags.unsqueeze(-1), concat, overlap)
        x1 = torch.where(mix_flags.unsqueeze(-1), mixed, x1)
        state = torch.where(mix_flags, torch.full_like(state, STATE_MIXED), state)
        # ponytail: batch-roll partner; padding tails dilute mixed labels slightly,
        # switch to dataset-level pair loading if label purity ever matters.
        return x1, state

    @torch.no_grad()
    def sample(
        self,
        duration: int,  # number of samples at target_sample_rate
        *,
        batch: int = 1,
        steps: int = 32,
        cfg_strength: float = 2.0,
        negative: str = "mixed",  # "mixed" | "null"
        sway_sampling_coef: float | None = None,
        timestep_mapping: str = "sway_sampling",
        timestep_power: float | None = None,
        shift: float = 1.0,
        use_epss: bool = True,
        seed: int | None = None,
    ):
        self.eval()
        device = self.device
        dtype = next(self.parameters()).dtype

        if negative == "mixed":
            neg_id = STATE_MIXED
        elif negative == "null":
            neg_id = STATE_NULL
        else:
            raise ValueError(f"Unknown negative: {negative}")
        state = torch.full((batch,), STATE_CLEAN, device=device, dtype=torch.long)
        neg_state = torch.full((batch,), neg_id, device=device, dtype=torch.long)

        requested = int(duration)
        aligned = int(math.ceil(requested / self.wav_frame_len) * self.wav_frame_len)

        if exists(seed):
            torch.manual_seed(seed)
        y0 = torch.randn(batch, aligned, device=device, dtype=dtype)

        def fn(t, x):
            def to_v(pred):
                if self.prediction == "flow":
                    return pred
                return self._x_to_v(pred, x, t)

            if cfg_strength < 1e-5:
                pred = self.transformer(x=x, state=state, time=t)
                return to_v(pred)

            # negative-sample classifier-free guidance:
            # push away from the speaker-inconsistent ("mixed") direction
            pred_cfg = self.transformer(x=x, state=state, time=t, cfg_infer=True, neg_state=neg_state)
            pred, neg_pred = torch.chunk(pred_cfg, 2, dim=0)
            v_pos = to_v(pred)
            v_neg = to_v(neg_pred)
            return v_pos + (v_pos - v_neg) * cfg_strength

        use_epss = use_epss and timestep_mapping == "sway_sampling"
        if use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=device, dtype=torch.float32)
        else:
            t = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)

        if timestep_mapping == "uniform":
            pass
        elif timestep_mapping == "sway_sampling":
            if sway_sampling_coef is not None:
                t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        elif timestep_mapping == "power":
            if timestep_power is None:
                raise ValueError("timestep_power must be provided when timestep_mapping='power'")
            t = t.pow(timestep_power)
        else:
            raise ValueError(f"Unknown timestep_mapping: {timestep_mapping}")

        if shift != 1.0:
            t = t / (t + shift * (1 - t))

        # WavTTS inference uses Euler ODE sampling.
        odeint_kwargs = dict(self.odeint_kwargs)
        odeint_kwargs["method"] = "euler"
        trajectory = odeint(fn, y0, t, **odeint_kwargs)

        out = trajectory[-1] / self.latents_scale
        return out[:, :requested], trajectory

    def forward(
        self,
        inp: float["b nw"],  # raw waveform
        *,
        lens: int["b"] | None = None,
    ):
        # handle raw waveform
        if inp.ndim != 2:
            raise ValueError(f"WavTTS expects raw waveform input [B, N], got {tuple(inp.shape)}")

        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device, dtype=torch.long)
        mask = lens_to_mask(lens, length=seq_len)

        # no-leaky mixing augmentation + per-sample state labels
        x1, state = self._mix_augment(inp, lens)
        if self.state_drop_prob > 0.0:
            drop = torch.rand(batch, device=device) < self.state_drop_prob
            state = torch.where(drop, torch.full_like(state, STATE_NULL), state)

        x1 = x1 * self.latents_scale

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = self._sample_time(batch, dtype=dtype, device=device)

        # sample xt (phi_t(x) in the paper)
        t = time.unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        raw_pred = self.transformer(x=φ, state=state, time=time, mask=mask, lens=lens)

        # interpret prediction
        if self.prediction == "flow":
            v_pred = raw_pred
            x_pred = φ + (1.0 - t) * v_pred
        elif self.prediction == "x_pred":
            x_pred = raw_pred
            v_pred = self._x_to_v(x_pred, φ, time)
        else:
            raise ValueError(f"Unknown prediction: {self.prediction}")

        # loss space
        if self.loss_space == "flow":
            loss = F.mse_loss(v_pred, flow, reduction="none")
        elif self.loss_space == "v":
            # v-loss (same target flow, but v_pred computed from x_pred) & use clamp_min
            denom = (1.0 - time).clamp_min(self.t_eps)
            while denom.ndim < φ.ndim:
                denom = denom.unsqueeze(-1)
            target = (x1 - φ) / denom
            loss = F.mse_loss(v_pred, target, reduction="none")
        elif self.loss_space == "x":
            loss = F.mse_loss(x_pred, x1, reduction="none")
        else:
            raise ValueError(f"Unknown loss_space: {self.loss_space}")

        loss = loss[mask]
        flow_loss = loss.mean()
        total_loss = flow_loss

        aux_mel_loss = torch.tensor(0.0, device=device)
        if self.use_aux_mel_loss and self.aux_mel_loss is not None:
            x1_flat_unscaled = x1 / self.latents_scale
            x1_pred_flat_unscaled = x_pred / self.latents_scale

            aux_mel_kwargs = {}
            if self.aux_mel_loss_masked:
                aux_mel_kwargs.update(
                    frame_mask=mask,
                    frame_lengths=lens,
                )

            aux_mel_loss = self.aux_mel_loss(
                x1_pred_flat_unscaled,
                x1_flat_unscaled,
                **aux_mel_kwargs,
            )
            total_loss = total_loss + aux_mel_loss

        loss_dict = {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "aux_mel_loss": aux_mel_loss,
        }

        return total_loss, loss_dict
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py -v`
Expected: 全部 PASS（含 Task 1 的 `test_import`——此時 `wavtts.model` 的 import 鏈已一致）。

- [ ] **Step 5: Commit**

```bash
git add src/wavtts/model/cfm.py tests/test_uncond_smoke.py
git commit -m "feat: rewrite CFM as unconditional model with no-leaky mixing augmentation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: CFM 負樣本 CFG 採樣測試

**Files:**
- Modify: `src/wavtts/model/cfm.py`（`sample` 已在 Task 3 寫入；此 task 只驗證，若測試揭露 bug 則修）
- Test: `tests/test_uncond_smoke.py`

**Interfaces:**
- Consumes: Task 3 的 `CFM.sample(...)`。

- [ ] **Step 1: 寫測試**

在 `tests/test_uncond_smoke.py` 追加：

```python
import pytest


@pytest.mark.parametrize(
    "cfg_strength,negative",
    [(2.0, "mixed"), (2.0, "null"), (0.0, "mixed")],
)
def test_sample_shapes(cfg_strength, negative):
    model = make_model()
    out, trajectory = model.sample(
        8000, batch=2, steps=2, cfg_strength=cfg_strength, negative=negative, seed=0
    )
    assert out.shape == (2, 8000)
    assert torch.isfinite(out).all()
    assert trajectory.shape[0] == 3  # steps+1 個時間點


def test_sample_duration_not_multiple_of_frame_len():
    model = make_model()
    out, _ = model.sample(8123, batch=1, steps=2, seed=0)
    assert out.shape == (1, 8123)


def test_sample_rejects_unknown_negative():
    model = make_model()
    with pytest.raises(ValueError):
        model.sample(8000, steps=2, negative="bogus")
```

- [ ] **Step 2: 跑測試**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py -v`
Expected: PASS。若 FAIL，修 `cfm.py` 的 `sample`（不改測試），直到 PASS。

- [ ] **Step 3: Commit**

```bash
git add tests/test_uncond_smoke.py src/wavtts/model/cfm.py
git commit -m "test: cover negative-sample CFG sampling paths

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: dataset 與 collate 改為 wav-only

**Files:**
- Modify: `src/wavtts/model/dataset.py`
- Test: `tests/test_uncond_smoke.py`

**Interfaces:**
- Produces:
  - `CustomDataset.__getitem__ -> {"wav": Tensor[nw]}`（不再有 `text`）。
  - `collate_fn(batch) -> {"wav": Tensor[b, nw], "wav_lengths": LongTensor[b]}`（只有這兩個 key）。
  - `load_dataset(dataset_name, dataset_type="CustomDataset", audio_type="raw", waveform_kwargs=dict())`——**移除 `tokenizer` 參數**；資料路徑為 `data/{dataset_name}`，既有含 tokenizer 後綴的資料目錄（如 `Emilia_ZH_EN_pinyin`）直接把完整目錄名寫進 `datasets.name`。

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_uncond_smoke.py` 追加：

```python
def test_collate_wav_only():
    from wavtts.model.dataset import collate_fn

    batch = [{"wav": torch.randn(1000)}, {"wav": torch.randn(800)}]
    out = collate_fn(batch)
    assert set(out.keys()) == {"wav", "wav_lengths"}
    assert out["wav"].shape == (2, 1000)
    assert out["wav_lengths"].tolist() == [1000, 800]
    assert torch.equal(out["wav"][1, 800:], torch.zeros(200))
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py::test_collate_wav_only -v`
Expected: FAIL（舊 collate 讀 `item["text"]`，KeyError）。

- [ ] **Step 3: 修改 `src/wavtts/model/dataset.py`**

三處修改：

(a) `CustomDataset.__getitem__` 的結尾（`while True` 迴圈與載入/重採樣邏輯不動，`row["text"]` 那行刪除）改為：

```python
        return {
            "wav": audio.squeeze(0),
        }
```

同時刪除迴圈內的 `text = row["text"]` 一行。

(b) `load_dataset` 整個函式改為：

```python
def load_dataset(
    dataset_name: str,
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    waveform_kwargs: dict = dict(),
) -> CustomDataset:
    """
    WavTTS only supports raw waveform datasets.
    dataset_type:
      - "CustomDataset": use default data path data/{dataset_name}
        (include any legacy tokenizer suffix, e.g. Emilia_ZH_EN_pinyin, in the name)
      - "CustomDatasetPath": pass the full path to a prepared dataset
    """

    print("Loading dataset ...")

    if audio_type != "raw":
        raise ValueError("WavTTS only supports raw waveform datasets; audio_type must be 'raw'.")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("wavtts").joinpath(f"../../data/{dataset_name}"))
        try:
            train_dataset = load_from_disk(f"{rel_data_path}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(train_dataset, durations=durations, **waveform_kwargs)

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(train_dataset, durations=durations, **waveform_kwargs)

    else:
        raise ValueError(f"Unsupported dataset_type for WavTTS wav-only training: {dataset_type}")

    return train_dataset
```

(c) `collate_fn` 整個函式改為：

```python
def collate_fn(batch):
    wavs = [item["wav"] for item in batch]
    wav_lengths = torch.LongTensor([w.shape[0] for w in wavs])
    max_wav_len = wav_lengths.max().item()

    padded_wavs = []
    for w in wavs:
        pad_len = max_wav_len - w.shape[0]
        padded_wavs.append(F.pad(w, (0, pad_len), value=0.0))

    return dict(
        wav=torch.stack(padded_wavs),  # [B, T_wav]
        wav_lengths=wav_lengths,
    )
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/wavtts/model/dataset.py tests/test_uncond_smoke.py
git commit -m "refactor: wav-only dataset and collate, drop tokenizer from load_dataset

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: trainer / train.py / config 整合

**Files:**
- Modify: `src/wavtts/model/trainer.py`
- Modify: `src/wavtts/train/train.py`
- Modify: `src/wavtts/configs/WavTTS.yaml`
- Test: `tests/test_uncond_smoke.py`

**Interfaces:**
- Consumes: `CFM.forward(inp, *, lens) -> (loss, loss_dict)`、`CFM.sample(duration, ...)`、`collate_fn` 的 `{"wav", "wav_lengths"}`、`load_dataset(dataset_name, waveform_kwargs=...)`。
- Produces: `accelerate launch src/wavtts/train/train.py --config-name WavTTS.yaml` 可組出完整訓練流程（不實跑訓練，用 config 實例化測試把關）。

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_uncond_smoke.py` 追加：

```python
def test_config_instantiates_model_and_trains_one_step():
    from importlib.resources import files as pkg_files

    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from wavtts.model import CFM

    cfg = OmegaConf.load(str(pkg_files("wavtts").joinpath("configs/WavTTS.yaml")))
    arch = OmegaConf.to_container(cfg.model.arch, resolve=True)
    arch.update(dim=64, depth=2, heads=2)  # 縮小以便 CPU 測試；參數名必須與 DiT 簽名一致
    cfm_kwargs = OmegaConf.to_container(cfg.model.cfm, resolve=True)
    model_cls = get_class(f"wavtts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(**arch, wav_frame_len=cfg.model.waveform.wav_frame_len),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **cfm_kwargs,
    )
    torch.manual_seed(0)
    loss, loss_dict = model(torch.randn(2, 16000) * 0.1)
    assert torch.isfinite(loss)
    loss.backward()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py::test_config_instantiates_model_and_trains_one_step -v`
Expected: FAIL（yaml 的 arch 還有 `text_dim` 等 DiT 已不認得的參數，cfm 區塊還有 `joint_cond_drop_prob`）。

- [ ] **Step 3: 改寫 `src/wavtts/configs/WavTTS.yaml`**

整檔取代為：

```yaml
hydra:
  run:
    dir: ckpts/${model.name}_${datasets.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}

datasets:
  name: Emilia_ZH_EN_pinyin  # prepared data dir name under data/ (keep any legacy suffix)
  batch_size_per_gpu: 19200  # 8 GPUs, 8 * 19200 = 153,600
  batch_size_type: frame  # frame | sample
  max_samples: 64  # max sequences per batch if use frame-wise batch_size. we set 32 for small models, 64 for base models
  num_workers: 16

optim:
  epochs: 686
  learning_rate: 7.5e-5
  num_warmup_updates: 20000  # warmup updates
  grad_accumulation_steps: 1  # note: updates = steps / grad_accumulation_steps
  max_grad_norm: 1.0  # gradient clipping
  bnb_optimizer: False  # use bnb 8bit AdamW optimizer or not

model:
  name: WavTTS_Uncond_Large  # model name
  backbone: DiT
  cfm:
    prediction: x_pred         # flow | x_pred
    loss_space: v         # flow | v   (v is typical for x_pred mode)
    t_sampling: logistic_normal      # uniform | logistic_normal
    P_mean: -0.8
    P_std: 0.8
    t_eps: 0.02
    latents_scale: 9.0
    use_aux_mel_loss: True
    aux_mel_loss_weight: 0.05
    sample_rate: 16000
    p_mix: 0.5              # prob a sample becomes a "mixed" negative
    p_concat: 0.5           # among mixed: concat (temporal switch) vs overlap
    mix_lambda_range: [0.3, 0.7]
    concat_point_range: [0.3, 0.7]
    concat_xfade_ms: 20
    state_drop_prob: 0.1    # drop state label to null
  arch:
    dim: 1152
    depth: 28
    heads: 16
    dropout: 0.0
    ff_mult: 4
    qk_norm: null  # null | rms_norm
    pe_attn_head: null
    attn_backend: torch  # torch | flash_attn
    attn_mask_enabled: False
    checkpoint_activations: False  # recompute activations and save memory for extra compute
    use_audio_proj: True
    audio_proj_dim: 1024
    audio_proj_hidden: 768
  waveform:
    target_sample_rate: 16000
    wav_frame_len: 160    # 160 @16k = 100Hz

ckpts:
  logger: wandb  # wandb | tensorboard | null
  log_samples: True  # infer random sample per save checkpoint
  save_per_updates: 50000  # save checkpoint per updates 50000
  keep_last_n_checkpoints: -1  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
  last_per_updates: 5000  # save last checkpoint per updates 5000
  exp_name: ${model.name}_${datasets.name}
  save_dir: ckpts/${model.name}_${datasets.name}
```

- [ ] **Step 4: 修改 `src/wavtts/train/train.py`**

整檔取代為：

```python
# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from wavtts.model import CFM, Trainer
from wavtts.model.dataset import load_dataset


os.chdir(str(files("wavtts").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("wavtts").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"wavtts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    cfm_kwargs = getattr(model_cfg.model, "cfm", {}) or {}

    exp_name = model_cfg.ckpts.exp_name
    wandb_resume_id = None

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, wav_frame_len=model_cfg.model.waveform.wav_frame_len),
        waveform_kwargs=model_cfg.model.waveform,
        **cfm_kwargs,
    )

    save_dir = model_cfg.ckpts.save_dir
    if os.path.isabs(save_dir):
        checkpoint_path = save_dir
    else:
        checkpoint_path = str(files("wavtts").joinpath(f"../../{save_dir}"))

    # init trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="WavTTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    train_dataset = load_dataset(model_cfg.datasets.name, waveform_kwargs=model_cfg.model.waveform)
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 修改 `src/wavtts/model/trainer.py`**

四處修改（其餘不動）：

(a) `train()` 開頭的 log_samples 區塊，刪除 `from wavtts.infer.utils_infer import cfg_strength, nfe_step, sway_sampling_coef` 一行，其餘保留：

```python
        if self.log_samples:
            target_sample_rate = train_dataset.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)
```

(b) 訓練迴圈內 batch 解包與 model 呼叫（原 `text_inputs = ...` 到 `loss, cond, pred, loss_dict = self.model(...)` 段）改為：

```python
                with self.accelerator.accumulate(self.model):
                    wav = batch["wav"]
                    wav_lengths = batch["wav_lengths"]

                    loss, loss_dict = self.model(wav, lens=wav_lengths)
                    self.accelerator.backward(loss)
```

同時刪除 `duration_predictor` 的 if 區塊（`# TODO. add duration predictor training` 那段——已無 text/duration 訓練情境，整段刪）。

(c) `Trainer.__init__` 簽名中刪除 `noise_scheduler: str | None = None` 與 `duration_predictor: torch.nn.Module | None = None` 兩個參數，及其對應的 `self.noise_scheduler = noise_scheduler`、`self.duration_predictor = duration_predictor` 與 wandb `model_cfg_dict` fallback dict 裡的 `"noise_scheduler": noise_scheduler,` 一行。

(d) log_samples 生成區塊（原 `infer_text = ...` 到兩個 `torchaudio.save` 段）改為：

```python
                    if self.log_samples and self.accelerator.is_local_main_process:
                        unwrap = self.accelerator.unwrap_model(self.model)

                        with torch.inference_mode():
                            gen_len = min(wav_lengths[0].item(), 10 * target_sample_rate)
                            generated, _ = unwrap.sample(
                                duration=gen_len,
                                steps=32,
                                cfg_strength=2.0,
                                sway_sampling_coef=-1.0,
                            )
                            gen_audio = generated.to(torch.float32).cpu()  # [1, N_gen]

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        self.model.train()
```

- [ ] **Step 6: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py -v`
Expected: 全部 PASS。

另外驗證 trainer 模組可載入（不跑訓練）：

Run: `.venv/bin/python -c "from wavtts.model.trainer import Trainer; from wavtts.train import train; print('ok')"`
Expected: 印出 `ok`（`wavtts.train.train` import 時會 `os.chdir`，屬預期）。

- [ ] **Step 7: Commit**

```bash
git add src/wavtts/model/trainer.py src/wavtts/train/train.py src/wavtts/configs/WavTTS.yaml tests/test_uncond_smoke.py
git commit -m "refactor: wire unconditional training through trainer, train.py, and config

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: 無條件採樣 CLI

**Files:**
- Create: `src/wavtts/infer/sample_uncond.py`
- Test: `tests/test_uncond_smoke.py`

**Interfaces:**
- Consumes: `CFM.sample(duration, *, batch, steps, cfg_strength, negative, seed)`、`WavTTS.yaml` 的 `model.arch`/`model.cfm`/`model.waveform` 結構。
- Produces: `main(argv: list[str] | None = None)` 可程式化呼叫；CLI 用法：
  `python src/wavtts/infer/sample_uncond.py --ckpt model.pt [--config path.yaml] [--duration_sec 5] [--num 4] [--steps 32] [--cfg_strength 2.0] [--negative mixed|null] [--seed N] [--out_dir samples_uncond]`

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_uncond_smoke.py` 追加：

```python
def test_sample_uncond_cli(tmp_path):
    from importlib.resources import files as pkg_files

    import torchaudio
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from wavtts.model import CFM

    # 縮小版 config
    cfg = OmegaConf.load(str(pkg_files("wavtts").joinpath("configs/WavTTS.yaml")))
    cfg.model.arch.dim = 64
    cfg.model.arch.depth = 2
    cfg.model.arch.heads = 2
    cfg_path = tmp_path / "tiny.yaml"
    OmegaConf.save(cfg, str(cfg_path))

    # 對應的隨機權重 checkpoint
    arch = OmegaConf.to_container(cfg.model.arch, resolve=True)
    model_cls = get_class(f"wavtts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(**arch, wav_frame_len=cfg.model.waveform.wav_frame_len),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **OmegaConf.to_container(cfg.model.cfm, resolve=True),
    )
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, str(ckpt_path))

    from wavtts.infer.sample_uncond import main

    out_dir = tmp_path / "out"
    main(
        [
            "--ckpt", str(ckpt_path),
            "--config", str(cfg_path),
            "--duration_sec", "0.5",
            "--num", "2",
            "--steps", "2",
            "--seed", "0",
            "--out_dir", str(out_dir),
        ]
    )
    wav_files = sorted(out_dir.glob("*.wav"))
    assert len(wav_files) == 2
    info = torchaudio.info(str(wav_files[0]))
    assert info.num_frames == 8000  # 0.5s @ 16k
    assert info.sample_rate == 16000
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py::test_sample_uncond_cli -v`
Expected: FAIL（`No module named 'wavtts.infer.sample_uncond'`）。

- [ ] **Step 3: 建立 `src/wavtts/infer/sample_uncond.py`**

```python
"""Minimal CLI for unconditional speech sampling with negative-sample CFG."""

import argparse
import os
from importlib.resources import files

import torch
import torchaudio
from hydra.utils import get_class
from omegaconf import OmegaConf

from wavtts.model import CFM


def load_model(ckpt_path: str, cfg, device: str) -> CFM:
    model_cls = get_class(f"wavtts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(
            **OmegaConf.to_container(cfg.model.arch, resolve=True),
            wav_frame_len=cfg.model.waveform.wav_frame_len,
        ),
        waveform_kwargs=OmegaConf.to_container(cfg.model.waveform, resolve=True),
        **OmegaConf.to_container(cfg.model.cfm, resolve=True),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "ema_model_state_dict" in ckpt:
        state = {
            k.replace("ema_model.", ""): v
            for k, v in ckpt["ema_model_state_dict"].items()
            if k not in ("initted", "update", "step")
        }
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="WavTTS unconditional sampling")
    parser.add_argument("--ckpt", required=True, help="checkpoint .pt path")
    parser.add_argument("--config", default=None, help="training yaml (default: packaged WavTTS.yaml)")
    parser.add_argument("--duration_sec", type=float, default=5.0)
    parser.add_argument("--num", type=int, default=4, help="number of samples to generate")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--negative", choices=["mixed", "null"], default="mixed")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", default="samples_uncond")
    args = parser.parse_args(argv)

    config_path = args.config or str(files("wavtts").joinpath("configs/WavTTS.yaml"))
    cfg = OmegaConf.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.ckpt, cfg, device)

    sr = cfg.model.cfm.sample_rate
    duration = int(args.duration_sec * sr)
    os.makedirs(args.out_dir, exist_ok=True)

    with torch.inference_mode():
        wavs, _ = model.sample(
            duration,
            batch=args.num,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            negative=args.negative,
            seed=args.seed,
        )

    for i in range(args.num):
        out_path = os.path.join(args.out_dir, f"sample_{i}.wav")
        torchaudio.save(out_path, wavs[i : i + 1].to(torch.float32).cpu(), sr)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑全部測試確認通過**

Run: `.venv/bin/python -m pytest tests/test_uncond_smoke.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/wavtts/infer/sample_uncond.py tests/test_uncond_smoke.py
git commit -m "feat: add unconditional sampling CLI with negative-sample CFG

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 完成標準

- `.venv/bin/python -m pytest tests/test_uncond_smoke.py -v` 全綠。
- `git log` 有 Task 1–7 各自的 commit。
- 未動到 Global Constraints 列出的 legacy 檔案。
