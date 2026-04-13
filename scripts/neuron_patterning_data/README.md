# GSE249572 — iPSC Neuron Patterning Multiome

Download scripts for multiome data from Fleck et al. (Science 2024,
DOI: 10.1126/science.adn6121) — "Human neuron subtype programming through
combinatorial patterning". Used in `immune_integration_v2`.

**Data location**: `/nfs/team283/vk7/sanger_projects/large_data/neuron_patterning_multiome/GSE249572/`

## Contents

| Component | Count | Format |
|-----------|-------|--------|
| RNA h5ad files | 3 samples | AnnData h5ad (raw counts) |
| ATAC fragment files | 3 samples | fragments.tsv.gz |

## Samples

| Sample ID | Description | RNA GSM | ATAC GSM |
|-----------|-------------|---------|----------|
| iPSC_prepat_d5_1 | NGN2 prepatterning multiome rep 1 | GSM7950270 (83.4 MB) | GSM7950271 (2.6 GB) |
| iPSC_prepat_d5_2 | NGN2 prepatterning multiome rep 2 | GSM7950272 (82.9 MB) | GSM7950273 (2.5 GB) |
| iPSC_ctrl | iPSC control multiome | GSM7950274 (14.0 MB) | GSM7950275 (2.8 GB) |

Total download: ~8.3 GB. Genome: hg38, processed with Cell Ranger ARC v2.0.0.

## Scripts

| File | Purpose |
|------|---------|
| `download_gse249572.py` | Downloads RNA h5ad + ATAC fragments from GEO FTP |
| `create_sample_mapping.py` | Generates `sample_mapping.csv` with per-sample file paths |
| `submit_download.sh` | bsub wrapper — submits download to long queue |

## Usage

```bash
# Dry run (check what will be downloaded)
python scripts/neuron_patterning_data/download_gse249572.py --dry-run

# Submit as LSF job
bash scripts/neuron_patterning_data/submit_download.sh
```

## External resources

- **GEO**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE249572
- **GitHub**: https://github.com/quadbio/iNeuron_patterning
- **Paper**: https://doi.org/10.1126/science.adn6121
