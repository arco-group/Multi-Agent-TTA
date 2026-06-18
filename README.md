# Multi-Agent Test-Time Adaptation for Robust 2D Medical Image-to-Image Translation

Public release of the codebase for the MICCAI 2026 paper on multi-agent TTA.

## Repository Layout

- `checkpoint/Ap_*`: predictor-agent checkpoints.
- `checkpoint/monitoring_agent/cyclegan/`: CycleGAN monitoring-agent checkpoints.
- `checkpoint/monitoring_agent/flow_matching/`: flow-matching monitoring-agent checkpoints.
- `cyclegan/`: CycleGAN branch.
- `flow_matching/`: flow-matching branch.
- `_MICCAI___2026____Extened_TTA.pdf`: paper PDF.

## Public Scripts

CycleGAN:

- `train.py`
- `test.py`
- `train_monitoring_agent.py`
- `test_monitoring_agent.py`
- `compute_loss_rec.py`
- `TTA_baseline.py`
- `TTA_ema.py`
- `TTA_rvt.py`

Flow matching:

- `train_flow_matching.py`
- `test_flow_matching.py`
- `train_monitoring_agent.py`
- `test_monitoring_agent.py`
- `compute_loss_rec.py`
- `TTA_baseline.py`
- `TTA_ema.py`
- `TTA_rvt.py`

Internal helper modules used by the public scripts are kept in place, but the release no longer includes the extra ablation, plotting, single-sample, preprocessing, or statistical-test scripts.

## Setup

- Start from `cyclegan/environment.yml` if you only need the CycleGAN branch.
- Add the MONAI / generative dependencies required by `flow_matching/` in the same environment if you want both branches.

## Checkpoints

- `Ap` checkpoints are already placed under `checkpoint/`.
- Monitoring-agent checkpoints are stored under `checkpoint/monitoring_agent/` with one folder per dataset/model pair.
- For flow matching, `--diff_ckpt` points to the predictor-agent checkpoint folder or prefix expected by the script.
- For CycleGAN, pass the task-model checkpoint directory and experiment name as expected by the original scripts.

## Notes

- Hardcoded local filesystem paths were removed from the public scripts.
- The old `rec_model` naming was replaced by `monitoring_agent` in the public entry points.
