# Multi-Agent TTA for 2D Medical Image Translation

[Irene Iele](https://scholar.google.com/citations?user=srLH7lkAAAAJ&hl=it&oi=ao)<sup>1</sup>, 
[Francesco Di Feola](https://scholar.google.com/citations?user=nzm0qagAAAAJ&hl=it)<sup>2</sup>, 
[Paolo Soda](https://scholar.google.com/citations?user=E7rcYCQAAAAJ&hl=it&oi=ao)<sup>1,2</sup>, 
[Rosa Sicilia](https://scholar.google.com/citations?user=d3yjHMMAAAAJ&hl=it&oi=ao)<sup>3</sup>,
[Matteo Tortora](https://matteotortora.github.io)<sup>4</sup>

<sup>1</sup>  University Campus Bio-Medico of Rome,
<sup>2</sup>  Umeå University,
<sup>4</sup> UniCamillus-Saint Camillus International University of
Health Sciences
<sup>4</sup>  University of Genoa
</div>
<img width="937" height="397" alt="method_tta_v2_page-0001" src="https://github.com/user-attachments/assets/ad22b910-aae6-40f9-9888-b11395210030" />

Public release of the code used in the MICCAI 2026 paper on Multi-Agent Test-Time Adaptation for 2D Medical Image Translation.

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

