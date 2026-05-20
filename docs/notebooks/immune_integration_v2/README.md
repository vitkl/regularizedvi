# docs/notebooks/immune_integration_v2/

v2 immune integration onboarding. Adds new multiome datasets (cancer + non-cancer + developmental + mucosal) into the v1 immune integration pipeline. See [.claude/plans/implement-these-steps-in-tranquil-parasol.md](../../../.claude/plans/implement-these-steps-in-tranquil-parasol.md) for the active sub-plan (§1 + §2) and [.claude/plans/incorporating-immune-cells-from-abundant-pixel.md](../../../.claude/plans/incorporating-immune-cells-from-abundant-pixel.md) for the parent plan (§1-§6).

## Phase 0 outputs

| File | Purpose |
|---|---|
| `annotation_harmonization_proposed.md` | Claude-authored cell-type vocab review (6 labelled datasets). User edits + copies approved rows to `annotation_harmonization.md`. |
| `annotation_harmonization.md` | Canonical SOURCE OF TRUTH (consumed by `_get_harmonization_maps()`). User-curated. |
| `metadata_harmonization_proposed.md` | Claude-authored column-rename review (all 11 v2 datasets → v1 STANDARD_OBS_COLS). User edits + copies approved rows to `metadata_harmonization.md`. |
| `metadata_harmonization.md` | Canonical SOURCE OF TRUTH for per-dataset obs column mappings. User-curated. |

## Phase 0 → Phase 1 gating

A labelled dataset whose section has not landed in canonical `annotation_harmonization.md` has no mapping → loader cannot resolve `harmonized_annotation` → fails fast. Similarly, missing `metadata_harmonization.md` entries flag at end-of-Phase-0 verification.
