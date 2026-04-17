# Sub-Plan 11: CLAUDE.md Update — Purpose-Based Covariate Guidance

**Parent plan**: `neighbourhood_correlation_plan.md` (CLAUDE.md Update section)

## Review Corrections (applied)

1. **Placement**: insert DIRECTLY AFTER the existing "5-Key Covariate Design (Core Innovation)" section (CLAUDE.md lines 14-21), so the `batch_key` fan-out note flows into the new purpose-based section.
2. **Clarify `batch_key` fan-out equivalence**: add a sentence:
   > "In `_model.py`, `batch_key` is semantically equivalent to `library_key` — it is the finest technical unit, fanning out to `ambient_covariate_keys`, `library_size_key`, `dispersion_key`."
3. **Mention new module file**: add line:
   > "New neighbourhood correlation metrics module: `src/regularizedvi/plt/_neighbourhood_correlation.py` uses `library_key`, `dataset_key`, `technical_covariate_keys` exclusively."
4. **Mention curated marker genes CSV**:
   > "Curated marker genes: `docs/notebooks/known_marker_genes.csv` (~192 genes, columns: gene, cell_type, lineage, category)."
5. **Format consistency**: use bulleted lists with backticks (matching existing style on lines 15-21). Replace bolded `**Why**:` / `**Hierarchy constraint**:` paragraph headers with short inline bullets:
   - Library = finest technical unit — per-library ambient/library_size/dispersion
   - Dataset = optional mid-level grouping of libraries, validated nested within
   - Technical = optional, non-hierarchical; may have multiple values specified
6. **Additional verification command**: `grep -n "library_key" CLAUDE.md` — confirms new key name landed.

## Primary file
`CLAUDE.md`

## Dependencies
None.

## Tasks

### 1. Add a new section to CLAUDE.md

Insert under an appropriate existing section (after "Key Architecture Gotchas" or near the "5-Key Covariate Design" section):

```markdown
## Purpose-Based Covariate Keys (new code convention)

New code should use **purpose-based covariate keys** instead of the generic `batch_key`:

- `library_key` — finest technical unit (sequencing run, lane, GEM well). Replaces `batch_key` in new metric/analysis code.
- `dataset_key` — groups of libraries from the same study. Optional.
- `technical_covariate_keys` — broad technical axes (embryo, experiment type, 10x kit). Optional, multiple allowed.

**Why**: `batch_key` is semantically ambiguous. The model's existing `ambient_covariate_keys`, `dispersion_key`, `library_size_key` are all finest-unit groupings — aligning with "library" terminology. `dataset_key` provides an explicit optional second level. `technical_covariate_keys` enables non-hierarchical axes.

**Hierarchy constraint**: Each `library_key` value must map to exactly one `dataset_key` value. Validate at setup.

**Graceful degradation**: When only `library_key` is provided, multi-level comparisons (cross-dataset, cross-technical) become empty but don't raise errors.

**Backward compatibility**: Existing code (`_model.py` `batch_key`, `_integration_metrics.py` `batch_key`) retains current semantics. New neighbourhood correlation metrics module (`_neighbourhood_correlation.py`) uses purpose-based keys exclusively.
```

### 2. Also update the "Active Experiments" or "Immune Integration Pipeline" section if helpful

Add pointer to the new plan reference:

```markdown
## Integration Assessment Plans (Active)
- Neighbourhood correlation metrics: `.claude/plans/neighbourhood_correlation_plan.md` — per-cell marker gene correlation with KNN neighbours, label-free. Stratified by library/dataset/technical.
```

## Verification
- `grep -n "purpose-based" CLAUDE.md` returns match
- Section renders correctly in markdown preview
