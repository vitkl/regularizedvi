# Plan: Verify & Fix imperative-riding-phoenix.md (ATAC BetaBinomial Decoder)

## Context

The plan `imperative-riding-phoenix.md` was created across a 700k+ token conversation (e6c65996 JSONL) with extensive user feedback (27 items, many with inline sub-corrections). A corrections file (`imperative-riding-phoenix-corrections.md`) was also produced. This review identifies remaining issues that need fixing before implementation.

**Source JSONL confirmed**: `e6c65996-1341-4cf4-b90f-a2b818e3e49f.jsonl` — ATAC theta investigation. Contains both `imperative-riding-phoenix.md` and `imperative-riding-phoenix-corrections.md` at end of conversation.

**Previous verification attempts** (31eae998, f045cb90, 085c35d8): All interrupted or unrelated — no findings were produced.

**Implementation status**: Zero implementation done. All TODO items in the plan are ahead of codebase.

---

## Verification Summary

### Plan coverage: 19/27 items fully covered, 8 gaps found

---

## CRITICAL ISSUES (3)

### Issue 1: Kon_Koff BetaBinomial parameterization is WRONG

**User specified** (Items 13, 17 — from the biophysical model in Item 1):
```
alpha = d * sensitivity * Kon
beta = d * Koff * (1 - sensitivity)
```
Mean = `s*Kon / (s*Kon + Koff*(1-s))` — sensitivity interacts differently with ON vs OFF rates.

**Plan currently uses** (Step 5a, lines 388-389):
```python
sp = px_rate.clamp(1e-8, 1 - 1e-8)  # = library * kon/(kon+koff) = s * p
alpha = d * sp
beta = d * (1 - sp)
```
Mean = `s * Kon/(Kon+Koff)` — collapses rates into probability first, then applies sensitivity.

**These are mathematically different.** The user's biophysical form has `alpha + beta = d*(s*Kon + Koff*(1-s))` while the plan's form has `alpha + beta = d`. The user's version captures the biophysical asymmetry: when sensitivity is low, Koff dominates more strongly.

**Note**: The user also wrote "or equivalently: alpha = d * sensitivity * p, beta = d * (1 - sensitivity * p)" in Item 17, but this is NOT actually equivalent to the Kon/Koff form. This was likely a shorthand error — the biophysical form (consistent with Item 1's original model) should be used for Kon_Koff.

**Fix**: In Step 5a Kon_Koff branch AND corrections doc Correction 4:
```python
elif _mod_decoder_type == "Kon_Koff":
    d = px_r  # concentration
    n = getattr(self, f"total_count_{name}")
    sensitivity = library[name]  # already sigmoid-mapped to (0,1)
    alpha = d * sensitivity * _kon
    beta = d * _koff * (1 - sensitivity)
    px = BetaBinomialDist(total_count=n, alpha=alpha, beta=beta)
```

Also update the decoder: Kon_Koff decoder should NOT multiply library * p inside px_rate. Instead, return `p = kon/(kon+koff)` as px_scale, and let generative() combine with sensitivity separately. This means for Kon_Koff, `px_rate` in the decoder should just be `p` (no library multiplication), and generative() builds alpha/beta directly from kon, koff, sensitivity, d.

### Issue 2: Feature scaling incompatible with probability-space decoders

**Current code** (`_multimodule.py` lines 1035-1049): After decoder returns `px_rate`, feature_scaling and modality_scale multiply it:
```python
px_rate = px_rate * _feature_scaling_factor  # softplus(param)/0.7, centered ~1.0
px_rate = px_rate * mod_scale
```

For probability/Kon_Koff decoders, `px_rate` is in (0,1). Multiplying by scaling factors can push it >1, making it invalid for BetaBinomial.

**The plan does not address this.** Step 5a clamps `sp = px_rate.clamp(1e-8, 1-1e-8)` which silently clips the effect — wrong.

**Fix**: Add validation in `__init__` (same pattern as additive_background):
```python
for name, dec_type in self.decoder_type_dict.items():
    if dec_type in ("probability", "Kon_Koff"):
        if name in self.feature_scaling:
            raise ValueError(
                f"Modality '{name}' uses decoder_type='{dec_type}' which operates in "
                f"probability space ∈ (0,1). feature_scaling (multiplicative counts) is "
                f"mathematically incompatible. Remove '{name}' from 'feature_scaling_covariate_keys'."
            )
```

### Issue 3: Missing attribution method updates

**User said** (Item 26): "Also use px_rate index in attributions."

The plan does not mention updating `get_modality_attribution()` for new decoder types. When decoder returns `(p, None, px_rate, px_dropout, px, kon, koff)`, the attribution method needs to know which index contains the observable mean (px_rate at index 2, same as other types — but needs verification).

**Fix**: Add to Step 7 or a new Step 7b: verify `get_modality_attribution()` uses `px_rate` (index 2) for all decoder types, and that the Jacobian computation handles variable return arity.

---

## MEDIUM ISSUES (5)

### Issue 4: `px_p_decoder` creation missing from __init__ section

Plan Step 3a shows secondary decoder reuse for Kon_Koff but does NOT show the new `self.px_p_decoder = nn.Linear(n_hidden, n_output)` needed for the `probability` decoder type in `__init__`. The corrected forward() references it.

**Fix**: Add to Step 3a:
```python
if decoder_type == "probability":
    self.px_p_decoder = nn.Linear(n_hidden, n_output)
```

### Issue 5: `variance_burst_size` fallback doesn't route to BB init

Current code (`_multimodel.py` lines 582-594): When `dispersion_init="variance_burst_size"`, non-burst modalities fall back to `compute_dispersion_init()`. If someone uses `variance_burst_size` with mixed decoders (RNA=burst, ATAC=probability), ATAC gets NB theta init instead of BB concentration init.

**Fix**: The `variance_burst_size` fallback should also check decoder type:
```python
# In variance_burst_size fallback for non-burst modalities:
if decoder_type_for_mod in ("probability", "Kon_Koff"):
    log_theta_init, _diag = compute_bb_dispersion_init(...)
else:
    log_theta_init, _diag = compute_dispersion_init(...)
```

### Issue 6: `compute_bb_dispersion_init()` needs full AnnData wrapper

Plan Step 1b shows only the numpy formula core. But Correction 5 calls it as `compute_bb_dispersion_init(self.adata.mod[mod_name], modality_genomic_width=..., ...)`. It needs a full function with AnnData/h5ad loading, Welford stats, and library CV² — similar to `compute_dispersion_init()`.

**Fix**: Document that `compute_bb_dispersion_init()` must accept AnnData, compute mean_g/var_g via the same chunked Welford as `compute_dispersion_init`, compute library CV², then apply the BB MoM formula.

### Issue 7: `global_max_robust` passing mechanism is fragile

Corrections doc uses `setattr` on adata_manager and `hasattr` in module init. This is brittle.

**Fix**: Pass `global_max_robust` as a dict alongside `library_log_means`/`library_log_vars` through the existing `_module_kwargs` mechanism.

### Issue 8: `regularise_dispersion_prior: 0.5` unexplained

User asked (Item 26): "What does this do? We may not want to create subexponential prior accidentally."

**Fix**: Add documentation to Step 2 explaining that `regularise_dispersion_prior` scales the containment prior KL weight. 0.5 means half-strength containment for BB concentration (weaker than RNA's 3.0) because ATAC has legitimately high overdispersion. Verify this doesn't create subexponential tails.

---

## MINOR ISSUES (5)

### Issue 9: Import location
User (Item 13): "import at the top of the file not here." Plan/corrections still show `from scvi.distributions import BetaBinomial` inside if-blocks.

### Issue 10: `math` vs `np`
User (Item 26): Plan Step 4b uses `math.log()` — should use `np`.

### Issue 11: Variable naming
User (Item 26): Corrections use `glm` and `ls` — should be `global_log_mean` and `log_sensitivity`.

### Issue 12: NaN handling in Welford
User (Item 12): Per-group computation when all values are 0. Plan mentions "weight by mean_g_k > 0" but lacks implementation detail. Should use pandas mean weighted by non-zero mean.

### Issue 13: kon/koff outputs not stored for downstream
Plan adds `_kon`/`_koff` accumulators but doesn't show them stored in a dict (like `burst_outputs` for burst_freq/burst_size). Needed for metrics and get_normalized_expression.

---

## ITEMS COVERED (no issues)

- Item 1: BetaBinomial model formulation → Steps 1-5
- Item 2: "I don't know yet" (NB vs Poisson) → Plan proceeds with BB
- Item 3: Rejection of editing parent plan → Plan is separate file
- Item 4: PoissonVI critique, SCALE/Bernoulli critique → Research file
- Item 5: Skill invocation → N/A (tooling)
- Item 6: "Deeper analysis needed" → Step 1 analysis
- Item 7: Fragment vs cut site, nucleosome units → Step 1c, NUCLEOSOME_UNIT_BP
- Item 8: "More investigation" → Research completed
- Item 9: Bone marrow + embryo datasets, extend _dispersion_init.py → Steps 1a, 1d
- Item 10: "Make full plan first" → Full plan created
- Item 11: (alpha,beta) parameterization, peak width as hyperparameter, clamping, decoder type pattern, burst_size reuse → Steps 2-6
- Item 14-15: Library handling corrections → Step 4c (SIMPLIFIED)
- Item 16: Detailed library flow corrections, sigmoid approach → Step 4c
- Item 19-21: Session continuation → N/A
- Item 22-24: Verification requests → N/A

---

## Recommended Fix Order

1. Fix Issue 1 (Kon_Koff parameterization) — mathematical correctness, most critical
2. Fix Issue 2 (feature_scaling validation) — prevents silent errors
3. Fix Issue 4 (px_p_decoder in __init__) — code won't work without it
4. Fix Issue 5 (variance_burst_size routing) — edge case but important for mixed models
5. Fix Issues 9-11 (naming, imports) — minor cleanup
6. Fix Issues 3, 6-8, 12-13 — documentation and robustness
7. Then proceed with implementation

---

## Files to Modify (in the plan, not codebase)

| File | What to fix |
|---|---|
| `imperative-riding-phoenix.md` Step 3a | Add `px_p_decoder` init for probability type |
| `imperative-riding-phoenix.md` Step 3b | Kon_Koff decoder: don't multiply library*p in px_rate |
| `imperative-riding-phoenix.md` Step 3c | Add feature_scaling validation |
| `imperative-riding-phoenix.md` Step 5a | Fix Kon_Koff alpha/beta to biophysical form |
| `imperative-riding-phoenix.md` Step 2 | Explain regularise_dispersion_prior=0.5 |
| `imperative-riding-phoenix.md` Step 6 | Fix variance_burst_size fallback routing |
| `imperative-riding-phoenix.md` Step 7 | Add attribution method verification |
| `imperative-riding-phoenix-corrections.md` Correction 4 | Fix Kon_Koff alpha/beta |
| Both files | Fix imports, naming (glm→global_log_mean, ls→log_sensitivity, math→np) |
