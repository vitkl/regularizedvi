# AD Brain 3-Region — GSE272082

Download scripts for multiome data from Alzheimer's disease brain 3-region snMultiome study.

**Data location**: `/nfs/team205/vk7/sanger_projects/large_data/ad_brain_3region/`

## Usage

```bash
# Dry run
python scripts/geo_download/download_multiome.py \
    --manifest data/gse272082_download_manifest.tsv \
    --output-dir /nfs/team205/vk7/sanger_projects/large_data/ad_brain_3region --dry-run

# Submit as LSF job
bash scripts/ad_brain_3region/submit_download.sh
```

## External resources

- **GEO**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272082
