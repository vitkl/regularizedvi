# Extract lung_smoking Seurat meta.data → CSV.
#
# Phase 0 step 0.4 of immune_integration_v2.
# Runs via Slurm wrapper `submit_lung_smoking_metadata.sh` in the `seurat` conda env
# (R 4.5.3 + Seurat 5.5.0 verified 2026-05-17).
#
# Args (env vars; passed by Slurm wrapper):
#   RDS_PATH  — input Seurat RDS (default: GSE241468 share_seur.rds)
#   CSV_OUT   — output CSV path

suppressPackageStartupMessages(library(Seurat))

rds_path <- Sys.getenv(
  "RDS_PATH",
  unset = "/nemo/lab/briscoej/home/users/kleshcv/large_data/lung_smoking/annotations/GSE241468_share_seur.rds"
)
csv_out <- Sys.getenv(
  "CSV_OUT",
  unset = "/nemo/lab/briscoej/home/users/kleshcv/large_data/lung_smoking/annotations/lung_smoking_meta.csv"
)

cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("Seurat version: %s\n", as.character(packageVersion("Seurat"))))
cat(sprintf("Reading: %s\n", rds_path))
flush.console()

t0 <- Sys.time()
seu <- readRDS(rds_path)
cat(sprintf("readRDS elapsed: %.1f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

meta <- seu@meta.data
cat(sprintf("meta.data rows=%d  cols=%d\n", nrow(meta), ncol(meta)))
cat(sprintf("meta.data columns: %s\n", paste(colnames(meta), collapse = ", ")))

dir.create(dirname(csv_out), showWarnings = FALSE, recursive = TRUE)
write.csv(meta, csv_out, row.names = TRUE)
cat(sprintf("Wrote %s (%.1f MB)\n", csv_out, file.info(csv_out)$size / 1e6))
