# Lung Smoking — GSE241468

Download scripts for multiome data from context-aware multiomics smoking response study.

**Data location**: `/nfs/team205/vk7/sanger_projects/large_data/lung_smoking/`

## Usage

```bash
# Dry run
python scripts/geo_download/download_multiome.py \
    --manifest data/gse241468_download_manifest.tsv \
    --output-dir /nfs/team205/vk7/sanger_projects/large_data/lung_smoking --dry-run

# Submit as LSF job
bash scripts/lung_smoking/submit_download.sh
```

## External resources

- **GEO**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE241468
