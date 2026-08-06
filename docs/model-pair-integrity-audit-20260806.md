# Gemma 4 Model-Pair Integrity Audit

Date: 2026-08-06  
Mode: full integrity audit adapted to source, configuration, documentation, and run artifacts  
Frozen pair: edge `google/gemma-4-E4B-it`; cloud `google/gemma-4-12b-it`

## Artifacts Checked

- All tracked Python executor-construction paths under `src/` and `scripts/`.
- Formal and local configuration templates, the historical locked local config,
  model locking, preflight, profiling, matrix execution, endpoint collection,
  and train/dev router-gate paths.
- README, CCFA state, execution/handoff guides, project brief, idea record,
  experiment plan, model setup in the manuscript, and model bibliography entry.
- Existing local manifests and cloud-result manifests, plus their evidence
  classification. Untracked `tmp_*` files are scratch diagnostics, are not
  formal entry points, and are not admissible evidence.

## Claim-Evidence Matrix

| Policy claim | Direct evidence | Status |
| --- | --- | --- |
| Formal edge is Gemma 4 E4B-IT | `config.py` constant and exact validator; formal template; negative unit test | enforced |
| Formal cloud is Gemma 4 12B-IT | `config.py` constant and exact validator; formal template; negative unit test | enforced |
| No alternative formal model roles | exact `{edge, cloud}` role-set check | enforced |
| Pair comparison uses matched decoding | explicit equality checks for token budget, sampling, temperature, top-p, seed, and thinking mode | enforced |
| Model identity survives result collection | formal run-context and manifest contain `model_ids` and immutable `model_revisions` | enforced for new runs |
| Router data cannot silently mix models | endpoint collector and train/dev gate require the exact pair and matching provenance | enforced for new runs |
| Service profile belongs to the pair | profile loader compares model ID and revision against the locked config | enforced |
| Historical non-Gemma outputs cannot support claims | `configs/evidence-policy.json` overrides stale evidence labels and lists exclusions | explicitly excluded |

## Numeric Consistency

- Software verification: 57 tests passed; source, scripts, and tests compiled.
- Existing WebShop-50 values remain diagnostic: edge/cloud average reward
  0.160/0.187. They were not changed or promoted.
- Existing formal Gemma paper-evidence manifests: 0. This is expected because
  the new pair has not yet been run with immutable model revisions.
- The old documentation's inconsistent test/audit counts were removed or
  updated; historical diagnostics are described without implying current-source
  equivalence.

## Citation Metadata and Context

- `gemma2026gemma4` now replaces the stale model-family citation in the model
  setup. Title, team author, year, arXiv identifier `2607.02770`, DOI, and URL
  match the Gemma 4 Technical Report record.
- The citation supports only the model-family identification. No new performance
  result was inserted into the manuscript in this audit.

## Severity Ledger

| Severity | Finding | Resolution |
| --- | --- | --- |
| critical | Formal template still selected a Qwen edge/cloud pair | fixed; exact Gemma pair is now required at runtime |
| high | Formal manifests recorded revisions but not model IDs | fixed for run context, manifest, endpoint provenance, and gate receipts |
| high | Legacy Qwen matrix contained a stale `paper_evidence=true` label | preserved bit-for-bit but superseded and excluded by project evidence policy |
| high | Historical local locked config could still be executed | blocked by default validation; allowed only inside non-evidence historical run audit |
| medium | Existing Gemma gate used mutable ModelScope `master` snapshots | excluded from evidence; cloud run must resolve immutable revisions first |
| medium | Local 4080 does not contain either Gemma checkpoint | intentional; run the tracked τ² smoke sequentially on 4090D and do not substitute another model |

## Safe Edit and Execution Rules

1. Do not hand-edit or overwrite historical `configs/local-smoke.locked.json`.
   Generate `configs/local-smoke-gemma4.locked.json` from the active template.
2. Run `python scripts/audit_model_pair.py` before model download, profiling,
   endpoint collection, or router fitting.
3. Do not reuse `cloud_results/service-profile.json`; it belongs to the excluded
   legacy matrix. Generate a new profile from the locked Gemma config.
4. Do not use a path ending in `snapshots/master` as formal provenance. Record
   the immutable upstream model revision in the locked config and manifests.
5. Pull future formal results into a new Gemma-specific directory; do not place
   them under the excluded legacy matrix directory.

## Verification

- `python scripts/audit_model_pair.py`: passed; 2 active templates checked,
  7 historical manifests excluded, 0 unexcluded paper-evidence manifests found.
- `python -m unittest discover -s tests -v`: 57/57 passed.
- `python -m compileall -q src scripts tests`: passed.
- Both active templates pass config validation with `--allow-unlocked`; execution
  still rejects them until all model/dataset/repository revisions are locked.

## Next CCFA Owner

`ccf-pipeline-orchestrator` should own the immutable-lock and 4080/4090D gate
handoff. After immutable revisions and a new service profile exist,
`ccf-experiment-designer` should own the paired train/dev evidence run.

## No-Invention Status

No experiment result was created, relabeled, or inferred. Historical result
files were not edited. This audit changes eligibility and runtime validation,
not measured outcomes.
