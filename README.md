# Multi-Agent TTA for 2D Medical Image Translation

Public release of the code used in the MICCAI 2026 paper on multi-agent test-time adaptation.

This repository contains only the two models used in the paper:

- `cyclegan/`
- `flow_matching/`

The release is trimmed to the scripts needed to train the task model, train the monitoring agent, compute reconstruction loss, and run the three adaptation strategies:

- `baseline`
- `ema`
- `rvt`

## Repository Layout

- `checkpoint/Ap_*`: predictor-agent checkpoints.
- `checkpoint/monitoring_agent/cyclegan/`: monitoring-agent checkpoints for the CycleGAN branch.
- `checkpoint/monitoring_agent/flow_matching/`: monitoring-agent checkpoints for the flow-matching branch.
- `cyclegan/`: CycleGAN branch code.
- `flow_matching/`: flow-matching branch code.
- `_MICCAI___2026____Extened_TTA.pdf`: paper PDF, ignored by git.

## Public Scripts

### CycleGAN branch

- `train.py`
- `test.py`
- `train_monitoring_agent.py`
- `test_monitoring_agent.py`
- `compute_loss_rec.py`
- `TTA_baseline.py`
- `TTA_ema.py`
- `TTA_rvt.py`

### Flow-matching branch

- `train_flow_matching.py`
- `test_flow_matching.py`
- `train_monitoring_agent.py`
- `test_monitoring_agent.py`
- `compute_loss_rec.py`
- `TTA_baseline.py`
- `TTA_ema.py`
- `TTA_rvt.py`

Internal helper modules remain in the tree because the public scripts import them directly.

## Checkpoints

The repository already includes the checkpoints needed to reproduce the public runs:

- predictor agent
- monitoring agent for CycleGAN
- monitoring agent for flow matching

The monitoring-agent checkpoints are organized by model and dataset.

## Setup

Use the CycleGAN environment as the base environment for the `cyclegan/` branch:

```bash
conda env create -f cyclegan/environment.yml
```

Then add the dependencies required by `flow_matching/` in the same environment, or manage that branch in a separate environment if you prefer a cleaner split.

## Notes

- Hardcoded private filesystem paths have been removed from the public scripts.
- The old `rec_model` naming is now exposed as `monitoring_agent` in the public options.
- Files related to paired models, colorization, single-image testing, preprocessing, plotting, and statistical tests were removed from the public release.
