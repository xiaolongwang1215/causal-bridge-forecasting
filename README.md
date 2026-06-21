# Hierarchical Causal Discovery and Bridge Forecasting for Greenhouse Leaf-Temperature Control

This repository contains the code used for the study on hierarchical AutoLag-PCMCI+ causal discovery and bridge forecasting for greenhouse leaf-temperature control.

## Requirements

* Python 3.10

## Repository Structure

```text
.
├── step0_clean_build/
├── step1_correlation/
├── step2_baseline_pcmciplus/
├── step3_autolag_pcmciplus/
├── step4_cbf/
├── step5_CBF_trigger_policy_eval/
└── step6_DeltaT_proxy_event_optimization/
```

### `step0_clean_build`

Data preprocessing, including data cleaning, timestamp alignment, variable construction, and preparation of the datasets used in subsequent analyses.

**Manuscript:** Materials and Methods — data preprocessing and dataset construction.

### `step1_correlation`

Correlation and lagged-correlation analyses used during preliminary data exploration.

This folder is not part of the main workflow in the manuscript. It is retained because it was used to inspect variable relationships before causal discovery.

### `step2_baseline_pcmciplus`

Baseline PCMCI+ causal-discovery scripts.

This folder is retained for comparison and verification. The main causal-discovery results in the manuscript are produced by the hierarchical AutoLag-PCMCI+ framework in `step3_autolag_pcmciplus`.

### `step3_autolag_pcmciplus`

Hierarchical AutoLag-PCMCI+ causal discovery, including lag-window selection, hierarchical constraints, causal-edge estimation, bootstrap stability analysis, and sensitivity analysis.

**Manuscript:** Section 2.1 and Section 3.1.

### `step4_cbf`

Causal Bridge Forecasting (CBF) for leaf-temperature prediction using the variables selected from the causal-discovery stage.

**Manuscript:** Section 2.2 and Section 3.2.

### `step5_CBF_trigger_policy_eval`

Replay-based evaluation of the control and trigger policies.

The results from this module are based on historical replay data and represent estimated resource-saving potential under the observed conditions.

**Manuscript:** Section 2.3 and Section 3.3.

### `step6_DeltaT_proxy_event_optimization`

DeltaT-based proxy early-warning analysis, including threshold sensitivity analysis and sensor-robustness evaluation.

DeltaT is calculated as:

```text
DeltaT = leaf temperature - air temperature
```

Both leaf-temperature and air-temperature measurements are required for this analysis.

**Manuscript:** Section 2.4 and Section 3.4.

## Main Workflow

The main workflow used in the manuscript is:

```text
step0_clean_build
        ↓
step3_autolag_pcmciplus
        ↓
step4_cbf
        ↓
step5_CBF_trigger_policy_eval
        ↓
step6_DeltaT_proxy_event_optimization
```

`step1_correlation` and `step2_baseline_pcmciplus` are supporting modules and are not required for the main workflow.

## Usage

1. Run the scripts in `step0_clean_build` to prepare the input data.
2. Run `step3_autolag_pcmciplus` to reproduce the causal-discovery analysis.
3. Run `step4_cbf` to reproduce the leaf-temperature forecasting experiments.
4. Run `step5_CBF_trigger_policy_eval` to reproduce the replay-based policy evaluation.
5. Run `step6_DeltaT_proxy_event_optimization` to reproduce the DeltaT-based early-warning analysis.

Before running a script, check the input and output paths defined in that file.

## Citation

Citation information will be added after the paper is accepted or published.