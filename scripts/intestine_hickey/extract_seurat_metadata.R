# Extract per-cell metadata (incl. cell-type labels) from the 8 Hickey/Becker
# clustered_*_object.rds Seurat objects in <annotations>/, write one CSV per
# compartment. Mirrors the pattern used for lung_smoking/GSE241468_share_seur.rds.
#
# Output columns: barcode, orig.ident, plus all columns in seu@meta.data.
#
# Usage:
#   Rscript scripts/intestine_hickey/extract_seurat_metadata.R \
#       /nemo/lab/briscoej/home/users/kleshcv/large_data/intestine_hickey/annotations
#
# Conda env: regularizedvi  (Seurat ≥ 4 required)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript extract_seurat_metadata.R <annotations_dir>")
}
annot_dir <- normalizePath(args[1], mustWork = TRUE)
cat("Annotations dir:", annot_dir, "\n")

suppressPackageStartupMessages({
  library(Seurat)
})

rds_files <- list.files(
  annot_dir,
  pattern = "^clustered_.*_object\\.rds$",
  full.names = TRUE
)
if (length(rds_files) == 0) {
  stop("No clustered_*_object.rds files found in ", annot_dir)
}
cat("Found", length(rds_files), "Seurat .rds files\n")

for (rds in rds_files) {
  compartment <- sub("^clustered_(.*?)_object\\.rds$", "\\1", basename(rds))
  out_csv <- file.path(annot_dir, paste0(compartment, "_metadata.csv"))

  if (file.exists(out_csv)) {
    cat("  [skip]", basename(rds), "->", basename(out_csv), "(already exists)\n")
    next
  }

  t0 <- Sys.time()
  cat("  [load]", basename(rds), "...\n")
  seu <- readRDS(rds)

  if (!inherits(seu, "Seurat")) {
    cat("    WARNING: not a Seurat object (class=", class(seu)[1], "); skipping\n", sep = "")
    rm(seu); gc(verbose = FALSE)
    next
  }

  md <- seu@meta.data
  md$barcode <- rownames(md)
  # Reorder so barcode is first
  md <- md[, c("barcode", setdiff(colnames(md), "barcode"))]

  cat(
    "    cells=", nrow(md),
    " cols=", ncol(md),
    " (", paste(colnames(md), collapse = ","), ")\n",
    sep = ""
  )

  write.csv(md, out_csv, row.names = FALSE, quote = TRUE)
  cat("  [write]", basename(out_csv), " (", format(file.info(out_csv)$size, big.mark = ","), " bytes, ",
      round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s)\n", sep = "")

  rm(seu, md); gc(verbose = FALSE)
}

cat("\nDone. Outputs written to:", annot_dir, "\n")
