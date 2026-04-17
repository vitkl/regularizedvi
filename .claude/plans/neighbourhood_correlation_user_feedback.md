# User Feedback on Plan (with line quotations)

## Integration penetration thresholds
> "Integration penetration: fraction of cells with >=1 (and >=5) cross-dataset neighbours, per dataset"
- 1 and 5 is quite small given the total N neighbours of 50 and relatively small N datasets.

## Terminology corrections
> `batch_key="dataset"`
- batch_key is not used. This is categorical covariate key.

> `ambient_covariate_keys=["batch"]`
- This should be library key not ambient — although they are after the same.

> `batch`
- Use library key for this to remove confusion — this is why purpose-based covariates were created. Add that purpose-based covariates should be used instead of batch_key to CLAUDE.md.

> "Cross-batch (xb): neighbours with different batch, same dataset / Cross-dataset (xd): neighbours with different dataset"
- Dataset layer should be optional but when it's provided the workflow needs to ensure that batch is contained only within one dataset.
- The word batch is confusing — this plan should replace this with a more descriptive term.

> "sb, xb, xd"
- Don't use abbreviations — this has to be readable.

## Normalization
> `log1p(count / total_count * 10000) — standard scanpy convention`
- This is wrong. We need: `normalised = count * (n_vars / total_count)`. This way average is 1.

> `Pearson correlation on log-transformed data`
- This should indeed be Pearson correlation but on normalised not log-transformed data.

## Gene selection
> "no compute_cluster_averages in regularizedvi — implement simple groupby mean"
- No, copy that function.

> "For each label set (level_1, level_2, level_3, level_4, harmonized_annotation), one-hot encode, compute per-group average"
- Actually this is handled by compute_cluster_averages — just use that and concatenate the result. Computing average per label and concatenating into one dataframe, then move to next step.

> "Specificity = mean_in_group / sum_of_means_across_groups (per gene)"
- This should simply be: `mean_per_gene_per_label / mean_per_gene_per_label.sum(0, keep_dims=True)`

> "Filter: absolute mean > 1 AND specificity > 0.1"
- This should be the hyperparameters to gene selection function.

> "Start with (1) for development, compute (2) as second pass"
- We need to see both.

> "Gene Selection" (general)
- This step should be applied to each dataset separately due to disagreements in how cells were labelled.

## Computation details
> "max-gap"
- What is the reasoning for this?

> "practical bimodality indicator for small n (~5-30 per mask)"
- We can have multimodal effects. Also cells can have many neighbours from incoming edges.

> "binarized for Approach A (each neighbour examined individually)"
- In this case correlation mean and median can be after weighting. Add the two weighted metrics.

## Missing neighbours / NaN handling
> "NaN (not -1 or 0) — both values misleadingly imply a measurement"
- I don't think this is correct. Superficially yes. But missing neighbours are simply an artifact of that we don't want to compute full dense cell-cell matrix. Not having a neighbour implies low connectivity and high distance. We are essentially stratifying cells by distance. Weighted metric deals with this correctly — no connection means connectivity weight = 0. Binary metric is simply binarised version which has 1 for connected cells and 0 for not connected.
- But we still need to track how many neighbours this cell had in each set.

> "Fraction of cells with zero cross-dataset neighbours"
- This should be for every mask.

## Dataset-specific cell types
> "Flag dataset-specific cell types (>90% cells from one dataset) — exclude from isolation scoring"
- Why is this excluded from isolation scoring?

> "isolation score (per-model headline metric) / Flag dataset-specific cell types"
- I don't think isolation score metric has this interpretation. I think it should be normalised relative to something else to reflect integration failure.

> "Cell type present in only 1 dataset → NaN cross-dataset is CORRECT / Flag using labels: if >90% of cells of type T come from one dataset, T is 'dataset-specific'"
- This is correct motivation but wrong implementation — we don't have labels for this analysis. The whole point is to define this analysis as a replacement for using labels. If labels are present scIB can be used — but useful labels require manual annotation on individual datasets. In this workflow we only use labels for marker sets — which is more robust.

## Failure mode framework — EXTENSIVE FEEDBACK

> "UNDER-INTEGRATED: bio fine but not bridging batches"
- High within-batch and low cross-batch/dataset is only under-integrated when everything is expected to be integrated.
- This is where dataset dimension adds a third dimension to failure modes and differs from stratification by library key.
- For stratification by library key this indeed signals under-integration — but for stratification by dataset:
  A) We need to condition the conclusion on stratification by library key (not fully clear how).
  B) We need to disregard this square for dataset because it could be simply a reflection of dataset-enriched cell types (this is exactly the problem with scIB metrics) — what we need here is a solution that extracts benchmark metric from comparing a subset of cells that are actually the same between datasets (what is bad is complete lack of between-dataset integration but some cells can still be perfectly integrated) — maybe this can be resolved by comparing distributions of per-cell metrics between alternative models (very important!).
  C) This number can be biologically informative (e.g., plotting it on UMAP highlights cell type specific combinations).
  D) Conditioning on whether other neighbours are also not integrated can give a more informative metric that separates different cell types from lack of broad integration (i.e., exclude cells when neighbours have high within-library correlation and low cross-dataset or no neighbours) — actually under-integration is when there are no neighbours — not where neighbours are not correlated = low cross-dataset correlation among neighbours suggests poor integration (positive failure) where a cell type is consistently paired incorrectly across datasets even when recognised correctly.
- This is why this table needs to have a dimension to handle no-neighbours case — because no neighbours differs from wrong neighbours.
- This changed neighbour-weighted computation too. Neighbour weighting is useful but mean doesn't carry over the distinction between closest cells agreeing on low correlation and having no connections across library or dataset stratification — this indeed has to be a separate axis.

> "Failure Mode Framework" (general)
- User typed quite a lot of useful thoughts, needs input exported and linked to the plan.

> "Within-batch correlation is the upper bound for what's achievable."
- This is a good way to think about this.

> "Key insight: within-batch as reference"
- Both within library and within dataset between batches is useful.
- For embryo data we are going to have different stratification between library and dataset with more complex relationships.

## Random baseline
> "compare marker gene correlation to random-gene-set correlation of same size — if random set also gives low correlation, the cell is genuinely noisy."
- Random baseline is useful — but sampling genes is wrong dimension — create random KNN with the same N neighbours per cell (variable N) — correlations across genes can be high. We need the comparisons to take this into account and this adds another dimension to 2x2xN classification.
- Need them presented as decision tree with columns showing which variable was added (very important representation change).

## Multiple marker gene groups
> "Low within-batch may be noisy data (not model failure). To distinguish: compare marker gene correlation to random-gene-set correlation"
- A more informative variant is to split markers by annotation level. This way we can compare full marker set to markers that differentiate T cell types in immune integration (especially very specific NEAT-seq types) or progenitor domains in the embryo. Yes, expand the list of gene sets!

## Failure mode assignment
> "Biological variance preservation: Positive/Negative failure / Batch integration: Positive/Negative failure / Variability..."
- Assign these descriptions to leaves on the decision tree.

> "Positive failure (false merge): low corr_mean..."
- This generally identifies low capacity or that the model is regularised in a way that prevents it from learning things correctly.

> "Negative failure (false split)"
- Either forces distinct populations (correlation within batch is higher than between batches → indicates these populations need to be analysed stratified or that model covariates need to be changed (very informative outcome)) or simply finds irrelevant cells semi-randomly on the other side (low correlation not just lower — this is where several marker gene groups will help).

> "NaN or very few cross-batch/dataset neighbours (isolation_frac high)"
- The problem is that this doesn't mean integration failure — it could mean dataset-specific cell types. We need to detect which cells are likely to be truly dataset specific.

> "High std or high gap → mixed neighbourhood (some correct, some wrong)"
- This could indicate merges of related cell types.

## Discrepancy metric
> "High/Low/Large + — Average masks heterogeneity — looks integrated but mixes distinct populations"
- This could be an indicator of debris/multiplets — pulled together by global similarity but containing signatures of many cell types.
- These might also have excessive N neighbours.

> "Rare: individuals match but average doesn't (distinct equally-correlated subtypes)"
- Gene set groups will likely be informative.

## Distribution comparisons
> "Compare median(corr_avg_xb) across batches — outlier batches = integration failures"
- Need to look at distributions. Almost complete lack of overlap between distributions suggests a measure for integration failure. This is also where non-connected cells come in. The challenge is distinguishing this from true biological non-overlap.
- We have to introduce additional "pure technical" covariate (on top of existing 2) between which we expect high integration (like "Embryo" for embryo). If batch distributions or specific cells are larger than this distribution that can indicate biological difference.

> "but batch-level distribution comparison can flag it indirectly"
- Exactly but this comparison needs a reference of what is expected to have integration. Maybe this is where between-library distributions for the same cell define a reference for reasonable correlations between datasets. This is where having wrong cells in the neighbour set becomes different to having no connections.

## Per-cell output columns
> "corr_mean, corr_median, corr_std, corr_cv, corr_gap, corr_discrepancy"
- Update this with my input (add weighted metrics).

> "Sum of log1p(CPM/10k)"
- This is not correct.

> "For each mask m..."
- We need derived correlation metrics where correlation is normalised by:
  1. Setting A: Correlation for same batch.
  2. Setting B: Correlation average across all neighbours.

## Representation
- Present failure modes as decision tree with columns showing which variable was added.
- When exporting input on the plan it needs to have the line quotations that the input was about.
