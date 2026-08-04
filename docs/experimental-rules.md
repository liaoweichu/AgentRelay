# Experimental Rules

These rules are mandatory for every AgentRelay experiment.

## Data

- Use only real, public datasets distributed by an official project source or through Hugging Face Datasets.
- Follow dataset types, scales, and sample sizes used by comparable peer-reviewed work whenever the resources permit.
- Preserve official train, validation, and test splits. If a benchmark exposes no test labels, use its official evaluation protocol.
- Do not fabricate tasks, rewrite samples to improve outcomes, or leak evaluation examples into prompts, routing labels, training, calibration, or model selection.
- Execution traces and router labels may be generated only by native model inference on the official training split. They are derived run artifacts, not replacement task data, and must include dataset revision, sample identifier, split, model revision, prompt hash, seed, and timestamp.

## Hardware stages

### Local verification

- Device: RTX 4080 Laptop.
- Allowed work: dependency checks, unit tests, small-batch native inference, pipeline debugging, and pilot profiling.
- Local measurements are diagnostic only and must not be reported as formal comparative results.

### Formal experiments

- Device: AutoDL RTX 4090D (24 GB).
- Required work: complete training runs, comparative baselines, ablations, robustness experiments, and quantitative evaluation.
- All reported main-paper numbers must originate from recorded RTX 4090D runs unless a table explicitly identifies an externally reported number and protocol mismatch.

## AutoDL persistence

Use `/root/autodl-tmp/AgentRelay` as the project data root:

```text
/root/autodl-tmp/AgentRelay/
  datasets/
  models/
  runs/
  results/
```

Repository code and lightweight configuration may live on the system disk, but pretrained weights, original/preprocessed datasets, checkpoints, run logs, and result files must live under this persistent data root.

## Implementation integrity

- Implement the declared method faithfully.
- Obtain every metric through native model inference and deterministic evaluation code.
- Record exact model and dataset revisions, seeds, configuration, command, environment, and hardware metadata.
- Prohibit result editing, cherry-picking without disclosure, implicit prompt cheating, test-set leakage, and manual metric optimization.
- Manuscript tables and figures must be generated from immutable recorded result files; never enter claimed values manually.
