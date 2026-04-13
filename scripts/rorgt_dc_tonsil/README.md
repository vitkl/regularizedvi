# RORγt+ DC-like Cells (Tonsil) — GSE247692

Download scripts for multiome data from RORγt+ DC-like cells tonsil study.

**Data location**: `/nfs/team205/vk7/sanger_projects/large_data/rorgt_dc_tonsil/`

## Usage

```bash
# Dry run
python scripts/geo_download/download_multiome.py \
    --manifest data/gse247692_download_manifest.tsv \
    --output-dir /nfs/team205/vk7/sanger_projects/large_data/rorgt_dc_tonsil --dry-run

# Submit as LSF job
bash scripts/rorgt_dc_tonsil/submit_download.sh
```

## External resources

- **GEO**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE247692
