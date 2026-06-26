# 🐾 Krea2 Character LoRA

> 120億パラメータの画像生成モデルを、消費者向けGPU 1台でキャラクター学習させる — LoRA ファインチューニング実験（Krea 2 編）

個人制作のオリジナルキャラクター「マサフィー」（LINEスタンプのレッサーパンダ）を、**約120億パラメータの Krea 2（Qwen-Image 系 MMDiT）** に LoRA ファインチューニングで学習させ、学習データに無い新しいポーズ・場面のイラストを生成できるようにした実験プロジェクトです。半精度の重み（約24GB）すら載らない12Bモデルの学習を、有料クラウドを使わず **自宅の NVIDIA GeForce RTX 3060（VRAM 12GB）1台** のみで成立させました。Stable Diffusion 1.5 で同じキャラを学習した前作 [masafee-lora](https://github.com/masafykun/masafee-lora) の続編・対構成です。

![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square) ![Krea 2](https://img.shields.io/badge/model-Krea%202%20(12B%20MMDiT)-8A2BE2?style=flat-square) ![LoRA](https://img.shields.io/badge/method-LoRA-34d399?style=flat-square) ![GPU](https://img.shields.io/badge/GPU-RTX%203060%2012GB-76b900?style=flat-square) [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-masafy%2Fkrea2--character--lora-FFD21E?style=flat-square)](https://huggingface.co/masafy/krea2-character-lora) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20838898.svg)](https://doi.org/10.5281/zenodo.20838898) [![ORCID](https://img.shields.io/badge/ORCID-0009--0000--7977--2756-A6CE39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0009-0000-7977-2756)

🔗 **[キャラクター「マサフィー」LINEスタンプ](https://store.line.me/stickershop/product/19794926/ja)** · 🤗 **[Hugging Face で学習済みLoRAを配布中](https://huggingface.co/masafy/krea2-character-lora)**

---

## 📸 スクリーンショット

最良モデル（epoch 10 / LoRA strength 0.8）で生成した、学習データに無い新規ポーズ：

![showcase](samples/pose_grid.png)

---

## ✨ 特徴

- **12GB で 12B モデルを学習** — bf16 では重み（約24GB）すら載らない Krea 2 を、**FP8 スケール量子化 + ブロックスワップ26 + PyTorch `expandable_segments`** の併用で RTX 3060（12GB）に収めた。断片化対策が無いと step 2 で OOM する。
- **消費者向けGPUのみで完結** — 有料クラウド不要。学習1回は **8時間23分**（約23.2秒/step、1,300 step、ピークVRAM 約11.9GB）。
- **少数データ学習** — オリジナルキャラの LINE スタンプ13点で学習。トリガーワード `masafee`。
- **RAW学習 → Turbo転移** — 蒸留前の RAW モデルで学習し、**8ステップ蒸留版 Turbo** 推論へ良好に転移する。
- **過学習の周波数非対称性** — 高周波の細部（目のキャッチライト）が低周波の全体形状より先に劣化する。この劣化は推論時に LoRA 強度とテキスト条件付けで**可逆的に回復**できる。
- **知見の論文化** — 手順・比較・結論を学術論文形式（日英 / LaTeX・PDF）にまとめて同梱。
- **再現可能** — 学習スクリプト・データセット・推論ワークフロー・provenance ログを公開。

---

## 🛠️ 技術スタック

| カテゴリ | 技術 |
|---|---|
| 画像生成モデル（学習） | Krea 2 RAW（約12B、single-stream MMDiT、flow matching） |
| 画像生成モデル（推論） | Krea 2 Turbo（8ステップ蒸留、GGUF Q5_K_M） |
| テキストエンコーダ / VAE | Qwen3-VL-4B / Qwen-Image VAE |
| 学習手法 | LoRA (rank 32, alpha 32, AdamW, lr 1e-4, 1,300 step) |
| 学習フレームワーク | Musubi Tuner (kohya-ss) |
| メモリ戦略 | FP8 scaled + block swap 26 + `expandable_segments` |
| 基盤 | PyTorch 2.6.0 + CUDA 12.4 / Python 3.11.15（`uv` で構築） |
| ハードウェア | NVIDIA GeForce RTX 3060 12GB |

---

## 📁 ディレクトリ構成

```
krea2-character-lora/
├── paper/                  論文（日英 × LaTeX / PDF）
│   ├── paper_ja.pdf         日本語版
│   ├── paper_en.pdf         英語版
│   └── figures/             損失曲線・同一性・エポックグリッド・目の回復
├── lora/                   LoRA 重み（Hugging Face で配布。lora/README.md 参照）
├── dataset/                学習に用いた13枚 + タグキャプション（トリガー: masafee）
├── scripts/                学習ランチャ・dataset 設定・ComfyUI 推論ワークフロー
│   ├── run_train.sh         学習実行
│   ├── dataset.toml         学習データ設定
│   ├── download_lora.sh     学習済みLoRA取得
│   └── inference_workflow_comfyui.json  ComfyUI 推論ワークフロー
├── samples/                生成サンプル（ポーズグリッド・同一性・目のキャッチライト比較）
└── logs/                   provenance・損失曲線・GPU テレメトリ
```

---

## 🚀 セットアップ

```bash
# 1. 学習ツール (Musubi Tuner) を取得
git clone https://github.com/kohya-ss/musubi-tuner.git

# 2. Python 3.11 環境を用意（uv 推奨。ディストリ標準が新しすぎる場合に有効）
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r musubi-tuner/requirements.txt   # 本実験は musubi-tuner @ 30c658c で実行

# 3. ベースモデルを配置（別途ダウンロード。ライセンス上ここでは再配布しない）
#    Krea 2 RAW / Qwen3-VL-4B / Qwen-Image VAE

# 4. latents と text-encoder 出力を precache → 学習を実行
bash scripts/run_train.sh
```

3060（12GB）で学習する場合は、起動引数 `--fp8_base --fp8_scaled --blocks_to_swap 26` と環境変数 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（これが無いと step 2 で VRAM OOM）が**必須**です。正確なバージョン・ハイパーパラメータは `logs/run_info.txt` と論文の付録に記録しています。

---

## 🤗 学習済みモデル

LoRA チェックポイント（epoch 6 / 8 / 10、各約448MB）は Hugging Face で配布しています：

> **https://huggingface.co/masafy/krea2-character-lora**

`scripts/download_lora.sh` または `huggingface_hub` ライブラリで取得できます。推奨は **epoch 10 を strength 0.8** で適用し、プロンプトに `big sparkling eyes with bright highlights` を加える設定です（目の輝きを回復しつつ全体デザインを維持。論文の Results 参照）。推論は Krea 2 Turbo（8 steps / cfg 1.0 / sampler `er_sde`）で行います。

---

## 📄 論文

実験の詳細・比較・知見は論文にまとめています。

- **日本語版:** [paper/paper_ja.pdf](paper/paper_ja.pdf)
- **English:** [paper/paper_en.pdf](paper/paper_en.pdf)

主な知見：(1) FP8 量子化＋ブロックスワップ＋`expandable_segments` の併用で、消費者向け12GB GPU 上での12Bモデル LoRA 学習が成立する、(2) 蒸留前 RAW で学習した LoRA は8ステップ蒸留版 Turbo へ良好に転移する、(3) 過学習は周波数非対称で、高周波の目のキャッチライトが先に劣化するが、推論時の LoRA 強度とプロンプトで可逆的に回復できる。

---

## 📚 引用 / Citation

このリポジトリを引用する場合は、以下の形式をお使いください（GitHub 右上の「Cite this repository」からも自動生成できます）。

```bibtex
@software{suzuki_krea2_character_lora_2026,
  author       = {Suzuki, Masato},
  title        = {{Krea2 Character LoRA: Single-GPU LoRA Fine-tuning of a
                   12B-Parameter Image Model for an Original Character}},
  year         = {2026},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.20838898},
  url          = {https://doi.org/10.5281/zenodo.20838898},
  orcid        = {0009-0000-7977-2756}
}
```

**DOI (永続アーカイブ):** [10.5281/zenodo.20838898](https://doi.org/10.5281/zenodo.20838898)

---

## 🎨 キャラクターについて

キャラクター「マサフィー」、および本リポジトリの学習画像・生成画像の著作権は著者（masafykun / Masato Suzuki）に帰属します。**キャラクターおよびその画像の二次利用・再配布・商用利用は許可していません。**

---

## ライセンス

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

- **コード**（`scripts/`・リポジトリのツール類）: **MIT** — [LICENSE](LICENSE) 参照。
- **論文・図・生成サンプル**: **CC BY 4.0**。
- **LoRA 重み・ベースモデル**: 本 LoRA は Krea 2 の派生物であり **Krea 2 Community License** に従います — [NOTICE.md](NOTICE.md) 参照。
- キャラクター「マサフィー」および学習画像は著者に帰属します（上記「キャラクターについて」参照）。

© 2026 masafykun (https://github.com/masafykun)
