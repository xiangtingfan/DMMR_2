# Seed4 3-Subject Comparison

This note keeps the baseline and SupCon runs aligned so their results are easy to compare.

## Goal

Compare the original DMMR pipeline against the new DMMR + SupCon pretraining variant on the same quick setting:

- dataset: `seed4`
- session: `1`
- held-out subjects: first `3`
- same optimizer defaults
- same pretraining / fine-tuning epoch counts unless manually overridden

## Run Commands

### 1. Baseline

```bash
python main_baseline_3subjects.py --dataset_name seed4 --session 1 --max_subjects 3 --index baseline_seed4_3subjects
```

Notes:

- `main_baseline_3subjects.py` is a non-invasive quick-entry copy so the original `main.py` remains untouched.

### 2. SupCon

```bash
python main_supcon.py --dataset_name seed4 --session 1 --max_subjects 3 --index supcon_seed4_3subjects
```

### 3. Optional shorter debug run

```bash
python main_supcon.py --dataset_name seed4 --session 1 --max_subjects 3 --epoch_fineTuning 50 --index supcon_seed4_3subjects_debug
```

## TensorBoard Paths

- Baseline: `data/session1/DMMR_BASELINE/seed4/baseline_seed4_3subjects`
- SupCon: `data/session1/DMMR_SUPCON/seed4/supcon_seed4_3subjects`

If you change `--way` or `--index`, the output path changes with it.

## What To Compare

### Final metrics

- Average test accuracy across the selected held-out subjects
- Standard deviation across the selected held-out subjects
- Per-subject best test accuracy

### Pretraining curves

- Baseline: `loss_pretrain`, `rec_loss`, `sim_loss`
- SupCon: `pretrain total loss`, `pretrain rec loss`, `pretrain adv loss`, `pretrain con loss`

### Fine-tuning curves

- Training accuracy
- Fine-tune classification loss
- Test accuracy

## Record Template

| Run Name | Model | Session | Held-out Subjects | beta | lambda_con | Temp | Avg Acc | Std | Subj0 | Subj1 | Subj2 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_seed4_3subjects | DMMR | 1 | 0,1,2 | 0.05 | 0 | - |  |  |  |  |  |  |
| supcon_seed4_3subjects | DMMR+SupCon | 1 | 0,1,2 | 0.05 | 0.1 | 0.07 |  |  |  |  |  |  |

## Suggested Ablation Order

1. Baseline quick comparison on subjects `0,1,2`
2. SupCon with `lambda_con=0.1`
3. SupCon with `lambda_con=0.05`
4. SupCon with `lambda_con=0.2`

## Current SupCon Defaults

- `beta=0.05`
- `lambda_con=0.1`
- `proj_hidden_dim=64`
- `proj_output_dim=32`
- `supcon_temperature=0.07`

## Current SupCon Logging

Per held-out subject, the new path writes these pretraining scalars:

- `train DMMR_SUPCON/pretrain total loss`
- `train DMMR_SUPCON/pretrain rec loss`
- `train DMMR_SUPCON/pretrain adv loss`
- `train DMMR_SUPCON/pretrain con loss`

And these fine-tuning scalars:

- `train DMMR_SUPCON/fine-tune cls loss`
- `train DMMR_SUPCON/train accuracy`
- `test DMMR_SUPCON/test acc`
