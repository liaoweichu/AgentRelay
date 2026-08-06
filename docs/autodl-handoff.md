# AutoDL RTX 4090D Handoff

The authoritative cloud-only handoff is:

- `docs/cloud-4090d-handoff-plan-20260806.md`

The local machine performs software and model-policy validation only. It does
not contain or run Gemma 4 E4B/12B. The first native model smoke, immutable
model lock, fresh service profile, τ² three-domain smoke, E4B precision
sensitivity, and τ² train/dev learnability gate all run sequentially on the
AutoDL RTX 4090D.

Do not use older archive-based handoff commands, the legacy Qwen service
profile, the legacy matrix, or ModelScope `snapshots/master` as formal model
provenance. Start after the model-policy changes are pushed to `main`; no Git
SHA or uploaded-file hash needs to be supplied or checked manually.
