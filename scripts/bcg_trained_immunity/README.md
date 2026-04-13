# BCG Trained Immunity — GSE295277 + GSE295308

Download scripts for multiome data from BCG vaccination trained immunity study.

**Data location**: `/nfs/team205/vk7/sanger_projects/large_data/bcg_trained_immunity/`

## Usage

```bash
# Dry run
python scripts/geo_download/download_multiome.py \
    --manifest data/gse295277_gse295308_download_manifest.tsv \
    --output-dir /nfs/team205/vk7/sanger_projects/large_data/bcg_trained_immunity --dry-run

# Submit as LSF job
bash scripts/bcg_trained_immunity/submit_download.sh
```

## External resources

- **GEO**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE295277
