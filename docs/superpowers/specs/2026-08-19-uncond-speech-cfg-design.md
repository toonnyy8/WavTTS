# WavTTS 無條件語音生成 + 混合語音負樣本 CFG — 設計文件

日期：2026-08-19
狀態：已與作者於對話中確認方向（從零訓練、batch-roll 混合、就地修改）

## 目標

把 WavTTS 從 zero-shot TTS 改造為**無條件純語音生成模型**：

- 移除 text conditioning 與 audio prompt（infilling）conditioning。
- 模型唯一的條件輸入是一個三值狀態標記：`clean=0`（單一語者乾淨語音）、
  `mixed=1`（兩位語者混合語音）、`null=2`（無條件）。
- 訓練時以 **no-leaky 混合增強**產生 mixed 樣本：整段波形（而非只有某區段）
  以等功率比例混入 batch 內另一條語音，標記與內容完全一致，不洩漏乾淨答案。
- 推論時以 CFG 做**負樣本引導**：正分支 `clean`、負分支 `mixed`（預設）或 `null`，
  `v = v_clean + w · (v_clean − v_neg)`，把生成推離「多語者混雜」方向，
  得到語者一致的單人語音。

## 非目標

- 不保留 TTS 功能。`src/wavtts/infer/infer_cli.py`、`utils_infer.py`、`eval/` 目錄
  與新模型不相容，**保留原檔但不維護**（不再被 import）。
- 不做 speaker prompt / voice cloning；模型沒有任何語者身分輸入。
- 不微調既有 checkpoint；從零訓練。

## 架構變更（就地修改）

### `src/wavtts/model/backbones/dit.py`

- 刪除 `TextEmbedding`、text cache（`text_cond`/`text_uncond`/`clear_cache`）、
  `get_input_embed` 的 text 邏輯。
- `InputEmbedding` 簡化為單一音訊分支：`x [b, n, wav_frame_len] → dim`
  （保留 `use_audio_proj` 兩層投影選項；刪除 `cond_proj` 與 text 串接），
  `ConvPositionEmbedding` 照舊。
- 新增 `self.state_embed = nn.Embedding(3, dim)`；forward 時
  `t = time_embed(time) + state_embed(state)`（state 為 `[b]` int tensor），
  即標準 DiT class-conditioning。
- `forward(x, state, time, mask=None, cfg_infer=False, neg_state=None, lens=None)`：
  `cfg_infer=True` 時打包正分支（傳入的 `state`）與負分支（`neg_state`）
  成 2b batch 一次 forward。
- waveform patchify（`wav_frame_len=160`、`_wav_to_tokens`/`_tokens_to_wav`）、
  DiT blocks、AdaLN、zero-init、`proj_out` 全部不變。

### `src/wavtts/model/cfm.py`

移除：`vocab_char_map`、text 處理、`frac_lengths_mask`、`rand_span_mask` infilling、
`audio_drop_prob` / `cond_drop_prob` / `joint_cond_drop_prob`、`mask_align_to` 對齊機制。

新增 CFM 參數（含預設值）：

| 參數 | 預設 | 意義 |
|---|---|---|
| `p_mix` | 0.5 | 每個樣本被做成 mixed 的機率 |
| `mix_lambda_range` | (0.3, 0.7) | 混合能量比 λ 的均勻取樣範圍 |
| `state_drop_prob` | 0.1 | 標記被 drop 成 `null` 的機率（保留純無條件採樣能力） |

#### 訓練 `forward(inp, lens=None)`

1. 對每個樣本抽 Bernoulli(`p_mix`) 決定是否混合。
2. 混合來源為 `partner = inp.roll(1, dims=0)`（batch-roll，零 IO 成本）；
   λ ~ U(`mix_lambda_range`)，等功率混合
   `x1 = sqrt(1−λ)·inp + sqrt(λ)·partner`。
3. 標記：混合樣本 `mixed`，其餘 `clean`；再以 `state_drop_prob` 逐樣本改為 `null`。
4. Flow matching 與現況相同（`prediction=x_pred`、`loss_space=v`、
   logistic-normal t 取樣、`latents_scale`），但 loss 遮罩改為整段有效長度
   （`lens_to_mask(lens)`），不再有 infilling span。
5. Aux mel loss 照舊，`frame_mask` 傳長度遮罩、`frame_lengths=lens`。

已知限制（接受）：partner 比本樣本短時，其 padding 零值區混不到干擾語音，
mixed 標記在該區段偏「乾淨」；dynamic batch sampler 依長度排序，同 batch 長度相近，
影響有限。`# ponytail: batch-roll mixing; switch to dataset-level pair loading if
label purity ever matters.`

#### 推論 `sample(...)`

新簽名：

```python
sample(duration, *, batch=1, steps=32, cfg_strength=2.0,
       negative="mixed",  # "mixed" | "null"
       sway_sampling_coef=None, timestep_mapping="sway_sampling",
       timestep_power=None, shift=1.0, use_epss=True, seed=None)
# duration: 生成樣本點數（int，16kHz）
```

- `y0 = randn(batch, duration)`，無任何 cond 填充/遮罩/裁切邏輯。
- ODE fn：`cfg_strength < 1e-5` 時單分支 `clean`；否則 packed forward
  正分支 `clean`、負分支 `negative` 對應的 state，
  `v = v_pos + cfg_strength · (v_pos − v_neg)`。
- timestep mapping（sway/EPSS/power/shift）與 Euler odeint 照舊。
- 回傳 `out / latents_scale` 與 trajectory。

### `src/wavtts/model/dataset.py`

- `CustomDataset.__getitem__` 只回傳 `wav`（資料集磁碟格式不變，text 欄位忽略）。
- `collate_fn` 只回傳 `wav`、`wav_lengths`。
- `load_dataset` 不再需要 tokenizer 參數（保留參數位置相容或直接刪除，採直接刪除）。

### `src/wavtts/model/trainer.py`

- 移除 text 相關批次欄位與 `duration_predictor` 呼叫中的 text。
- `self.model(wav, lens=wav_lengths)`。
- `log_samples`：改為無條件生成——取 batch 第一條的長度（上限 10 秒）生成一條，
  存 `update_{n}_gen.wav`；不再存 ref。不再 import `utils_infer` 的常數，
  直接用固定值（`steps=32`、`cfg_strength=2.0`、`sway_sampling_coef=-1.0`）。

### `src/wavtts/train/train.py`

- 刪除 tokenizer/vocab 邏輯；`model_cls(**arch, wav_frame_len=...)`。

### `src/wavtts/configs/WavTTS.yaml`（就地修改）

- 刪除 `tokenizer`、`tokenizer_path`、arch 的 `text_dim`/`text_mask_padding`/
  `conv_layers`（text conv）欄位。
- `cfm` 新增 `p_mix: 0.5`、`mix_lambda_range: [0.3, 0.7]`、`state_drop_prob: 0.1`；
  刪除 `joint_cond_drop_prob`。
- 其餘超參不變。

### 新增 `src/wavtts/infer/sample_uncond.py`

極簡 CLI：`--ckpt --duration_sec 5 --num 4 --steps 32 --cfg_strength 2.0
--negative mixed --seed --out_dir`。載入 checkpoint（EMA state dict 優先）、
建模型、`sample()`、`torchaudio.save`。

## 驗證

`tests/test_uncond_smoke.py`（不依賴資料集與 checkpoint）：

- 建一個縮小的 DiT（dim=64, depth=2）+ CFM。
- 隨機 `[4, 16000]` batch（含不等長 lens）跑 `forward` + `backward`，
  assert loss 有限、梯度存在。
- `sample(duration=8000, steps=2)`，assert 輸出 shape `[batch, 8000]` 且有限，
  分別測 `negative="mixed"` 與 `"null"` 與 `cfg_strength=0`。

## 風險與備註

- CFG 負分支用 `mixed` 是本設計的核心假設：`(v_clean − v_mixed)` 方向近似
  「單語者純度」梯度。若實驗顯示引導過強造成 artifacts，可下調 `cfg_strength`
  或改 `negative="null"` 做傳統 CFG 對照。
- `p_mix=0.5` 給負分支足夠的訓練訊號；若 clean 品質受影響可降至 0.3。
- 混合用等功率係數（`sqrt(1−λ)`, `sqrt(λ)`）避免破音；不做響度對齊
  （Emilia 已大致正規化）。
