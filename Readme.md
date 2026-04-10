# DMMR

Official PyTorch implementation for the AAAI 2024 paper:
`DMMR: Cross-Subject Domain Generalization for EEG-Based Emotion Recognition via Denoising Mixed Mutual Reconstruction`

[Paper link](https://ojs.aaai.org/index.php/AAAI/article/view/27819)

## Overview

This repository now contains:

- The original DMMR training pipeline
- Quick baseline scripts for partial-subject comparisons
- A supervised contrastive pretraining variant (`SupCon`)
- T-SNE utilities for baseline vs. SupCon feature visualization

## Environment

Install the dependencies with:

```bash
pip install -r requirements.txt
pip install tensorboard
```

If you use a Conda environment, activate it before running the commands below.

## Datasets

The public SEED and SEED-IV datasets can be downloaded from:
[https://bcmi.sjtu.edu.cn/home/seed/index.html](https://bcmi.sjtu.edu.cn/home/seed/index.html)

This code expects `.mat` feature files, one file per subject, under a session-specific folder.

The default local paths are:

- `SEED`: `H:/SEED/feature_for_net_session{session}_LDS_de`
- `SEED-IV`: `H:/SEED-IV/feature_for_net_session{session}_LDS_de`

If your dataset is stored elsewhere, override the path from the command line:

```bash
python main.py --dataset_name seed4 --session 1 --seed4_path "D:/your/path/session{session}"
```

## Main Files

- [main.py](./main.py): original DMMR entry point
- [main_baseline_3subjects.py](./main_baseline_3subjects.py): quick baseline comparison on a subset of held-out subjects
- [main_supcon.py](./main_supcon.py): DMMR + supervised contrastive pretraining
- [train.py](./train.py): original DMMR training code
- [train_supcon.py](./train_supcon.py): SupCon training code
- [T-SNE/generatePlotByTSNE.py](./T-SNE/generatePlotByTSNE.py): original T-SNE script
- [T-SNE/generate_method_tsne.py](./T-SNE/generate_method_tsne.py): method-aware T-SNE generation
- [T-SNE/make_clean_comparison_tsne.py](./T-SNE/make_clean_comparison_tsne.py): clean baseline vs. SupCon comparison panels

## Run The Original DMMR

Run the original full experiment:

```bash
python main.py --dataset_name seed4 --session 1 --index 1
```

Examples:

```bash
python main.py --dataset_name seed3 --session 1 --index 0
python main.py --dataset_name seed4 --session 1 --index 1
```

Outputs:

- TensorBoard logs under `data/session{session}/...`
- Saved checkpoints under `model/...`

## Run A Quick Baseline Comparison

For a quick baseline comparison on the first 3 held-out subjects:

```bash
python main_baseline_3subjects.py --dataset_name seed4 --session 1 --max_subjects 3 --index baseline_seed4_3subjects
```

Useful arguments:

- `--max_subjects`: how many held-out subjects to run
- `--index`: experiment subfolder name
- `--epoch_fineTuning`: reduce this for faster debugging

Example debug run:

```bash
python main_baseline_3subjects.py --dataset_name seed4 --session 1 --max_subjects 3 --epoch_fineTuning 50 --index baseline_seed4_debug
```

## Run DMMR + SupCon

Run the SupCon pretraining variant:

```bash
python main_supcon.py --dataset_name seed4 --session 1 --max_subjects 3 --index supcon_seed4_3subjects
```

Important arguments:

- `--lambda_con`: weight for supervised contrastive loss
- `--supcon_temperature`: temperature in the SupCon loss
- `--proj_hidden_dim`: projection head hidden dimension
- `--proj_output_dim`: projection head output dimension
- `--start_subject`: start from a specific held-out subject index
- `--skip_existing`: skip subjects whose checkpoint files already exist

Examples:

Quick 3-subject run:

```bash
python main_supcon.py --dataset_name seed4 --session 1 --max_subjects 3 --index supcon_seed4_3subjects
```

Continue a full 15-subject run after subject `0` is already finished:

```bash
python main_supcon.py --dataset_name seed4 --session 1 --index supcon_seed4_3subjects --max_subjects 14 --start_subject 1 --skip_existing
```

Smaller contrastive weight:

```bash
python main_supcon.py --dataset_name seed4 --session 1 --max_subjects 3 --lambda_con 0.05 --index supcon_seed4_lambda005
```

## TensorBoard Logging

Start TensorBoard with:

```bash
tensorboard --logdir data
```

Original DMMR writes logs such as:

- `train DMMR/loss`
- `train DMMR/train accuracy`
- `test DMMR/test acc`

SupCon writes additional pretraining curves:

- `train DMMR_SUPCON/pretrain total loss`
- `train DMMR_SUPCON/pretrain rec loss`
- `train DMMR_SUPCON/pretrain adv loss`
- `train DMMR_SUPCON/pretrain con loss`

## T-SNE Visualization

### 1. Original script

The original T-SNE script is:

```bash
python T-SNE/generatePlotByTSNE.py --dataset_name seed4 --session 1 --one_subject 0 --model_way DMMR/seed4 --model_index 1
```

This generates:

- `origin_subject.jpg`
- `origin_label.jpg`
- `pretrain_subject.jpg`
- `pretrain_label.jpg`
- `tune_subject.jpg`
- `tune_label.jpg`

### 2. Method-aware T-SNE

To generate T-SNE for a specific method and held-out subject:

Baseline example:

```bash
python T-SNE/generate_method_tsne.py --dataset_name seed4 --session 1 --one_subject 1 --method baseline --model_way DMMR/seed4 --model_index 1 --output_dir T-SNE/plot/seed4/session1/baseline/subject2
```

SupCon example:

```bash
python T-SNE/generate_method_tsne.py --dataset_name seed4 --session 1 --one_subject 1 --method supcon --model_way DMMR_SUPCON/seed4 --model_index supcon_seed4_3subjects --output_dir T-SNE/plot/seed4/session1/supcon_seed4_3subjects/subject2
```

Only draw a subset of views:

```bash
python T-SNE/generate_method_tsne.py --dataset_name seed4 --session 1 --one_subject 8 --method supcon --model_way DMMR_SUPCON/seed4 --model_index supcon_seed4_3subjects --views tune_subject,tune_label --output_dir T-SNE/plot/seed4/session1/supcon_seed4_3subjects/subject9_compact
```

Supported `--views` values:

- `origin_subject`
- `origin_label`
- `pretrain_subject`
- `pretrain_label`
- `tune_subject`
- `tune_label`

### 3. Clean comparison panels

To generate the clean comparison panels used for subject 2 vs. subject 9:

```bash
python T-SNE/make_clean_comparison_tsne.py
```

This produces:

- `T-SNE/plot/seed4/session1/comparison_clean/origin_comparison_grid.jpg`
- `T-SNE/plot/seed4/session1/comparison_clean/pretrain_comparison_grid.jpg`
- `T-SNE/plot/seed4/session1/comparison_clean/tune_comparison_grid.jpg`

## Ablation Studies

Run:

```bash
python ablation/witoutMix.py
python ablation/withoutNoise.py
python ablation/withoutBothMixAndNoise.py
```

## Other Noise Injection Methods

Run:

```bash
python noiseInjectionMethods/maskChannels.py
python noiseInjectionMethods/maskTimeSteps.py
python noiseInjectionMethods/channelsShuffling.py
python noiseInjectionMethods/Dropout.py
```

## Result Notes

- The repository ignores large generated files such as `model/`, `data/`, `runs/`, and `T-SNE/plot/` by default.
- If you want to reproduce existing reported numbers, make sure the dataset paths and session settings match the original experiments.
- For fair baseline vs. SupCon comparison, compare per-subject `best test acc` rather than the last training epoch.

## Citation

If you found this repository useful for your research, please cite:

```bibtex
@inproceedings{wang2024dmmr,
  title={DMMR: Cross-Subject Domain Generalization for EEG-Based Emotion Recognition via Denoising Mixed Mutual Reconstruction},
  author={Wang, Yiming and Zhang, Bin and Tang, Yujiao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={628--636},
  year={2024}
}
```

We also thank these repositories for helpful reference code:

- [MS-MDA](https://github.com/VoiceBeer/MS-MDA)
- [DANN](https://github.com/fungtion/DANN)
