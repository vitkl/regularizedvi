# BACH2/AP-1 Gut T Cells — GSE299348

Download scripts for multiome data from BACH2/AP-1 gut T cell study.

**Data location**: `/nfs/team205/vk7/sanger_projects/large_data/bach2_ap1_gut_tcells/`

## Usage

```bash
# Dry run
python scripts/geo_download/download_multiome.py \
    --manifest data/gse299348_download_manifest.tsv \
    --output-dir /nfs/team205/vk7/sanger_projects/large_data/bach2_ap1_gut_tcells --dry-run

# Submit as LSF job
bash scripts/bach2_ap1_gut_tcells/submit_download.sh
```

## External resources

- **GEO**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE299348
