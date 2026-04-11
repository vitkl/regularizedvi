#!/usr/bin/env Rscript
# Extract metadata from HTAN WUSTL pan-cancer multiome Seurat .rds files.
#
# Produces a single flat CSV with cell-level annotations matching the format
# used by existing immune datasets (e.g. GSE244831_cell_annotations.csv).
#
# Each .rds file is a Seurat object (~1-3 GB) containing a peak or gene
# assay, reductions (lsi/umap), and meta.data with cell_type / seurat_clusters /
# QC columns. We use SeuratObject directly (no full Seurat load) to extract
# meta.data quickly.
#
# Usage:
#   Rscript scripts/pan_cancer_data/convert_rds_annotations.R <input_dir> <output_csv> [modality]
#
# Examples:
#   Rscript scripts/pan_cancer_data/convert_rds_annotations.R \
#     /nfs/team205/vk7/sanger_projects/large_data/pan_cancer_multiome/level4/atac \
#     /nfs/team205/vk7/sanger_projects/large_data/pan_cancer_multiome/annotations/pan_cancer_multiome_atac_annotations.csv \
#     atac
#
#   Rscript scripts/pan_cancer_data/convert_rds_annotations.R \
#     /nfs/team205/vk7/sanger_projects/large_data/pan_cancer_multiome/level4/rna \
#     /nfs/team205/vk7/sanger_projects/large_data/pan_cancer_multiome/annotations/pan_cancer_multiome_rna_annotations.csv \
#     rna

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript convert_rds_annotations.R <input_dir> <output_csv> [modality]")
}

input_dir <- args[1]
output_csv <- args[2]
modality <- if (length(args) >= 3) args[3] else "unknown"

dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(SeuratObject))

rds_files <- list.files(input_dir, pattern = "\\.rds$", full.names = TRUE)
cat(sprintf("Found %d .rds files in %s\n", length(rds_files), input_dir))

# Column names vary per file (e.g. nCount_peaksMACS2 vs nCount_pancan),
# so we take the union of ALL meta.data columns across files.

extract_one <- function(rds_path) {
  fname <- basename(rds_path)
  cat(sprintf("[%s] %s ... ", format(Sys.time(), "%H:%M:%S"), fname))
  t0 <- Sys.time()

  obj <- tryCatch(readRDS(rds_path),
                  error = function(e) {
                    cat(sprintf("READ ERROR: %s\n", e$message))
                    return(NULL)
                  })
  if (is.null(obj)) return(NULL)

  meta <- tryCatch(as.data.frame(obj@meta.data),
                   error = function(e) {
                     cat(sprintf("META ERROR: %s\n", e$message))
                     return(NULL)
                   })
  if (is.null(meta) || nrow(meta) == 0) {
    rm(obj); gc(verbose = FALSE)
    return(NULL)
  }

  meta$barcode <- rownames(meta)
  meta$source_file <- fname

  # Derive cancer_type from filename prefix if present (ATAC: CANCER_SAMPLE.rds)
  cancer_type <- NA_character_
  stem <- tools::file_path_sans_ext(fname)
  if (grepl("_", stem)) {
    cancer_type <- strsplit(stem, "_")[[1]][1]
  }
  meta$cancer_type <- cancer_type

  # Try to extract UMAP coordinates
  umap_coords <- tryCatch({
    reductions <- slot(obj, "reductions")
    if (!is.null(reductions) && "umap" %in% names(reductions)) {
      as.data.frame(slot(reductions$umap, "cell.embeddings"))
    } else NULL
  }, error = function(e) NULL)

  if (!is.null(umap_coords) && nrow(umap_coords) == nrow(meta)) {
    colnames(umap_coords) <- c("UMAP_1", "UMAP_2")[seq_len(ncol(umap_coords))]
    meta <- cbind(meta, umap_coords[rownames(meta), , drop = FALSE])
  }

  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("%d cells, %.1fs\n", nrow(meta), dt))

  rm(obj); gc(verbose = FALSE)
  return(meta)
}

all_meta <- list()
n_ok <- 0
n_fail <- 0

for (rds_path in rds_files) {
  meta <- extract_one(rds_path)
  if (!is.null(meta)) {
    all_meta[[basename(rds_path)]] <- meta
    n_ok <- n_ok + 1
  } else {
    n_fail <- n_fail + 1
  }
}

cat(sprintf("\nExtracted metadata from %d / %d files (%d failed)\n",
            n_ok, length(rds_files), n_fail))

if (length(all_meta) == 0) {
  stop("No metadata extracted.")
}

# Find union of columns across all files
all_cols <- unique(unlist(lapply(all_meta, colnames)))
cat(sprintf("Union of columns (%d):\n", length(all_cols)))
print(all_cols)

# Bind all into one data frame (filling missing columns with NA)
fill_missing <- function(df, cols) {
  for (c in cols) {
    if (!(c %in% colnames(df))) df[[c]] <- NA
  }
  df[, cols, drop = FALSE]
}
combined <- do.call(rbind, lapply(all_meta, fill_missing, cols = all_cols))
cat(sprintf("\nCombined: %d cells x %d columns\n", nrow(combined), ncol(combined)))

write.csv(combined, output_csv, row.names = FALSE)
cat(sprintf("Written: %s\n", output_csv))

# Summary
cat("\nCell types:\n")
if ("cell_type" %in% colnames(combined)) {
  print(sort(table(combined$cell_type), decreasing = TRUE))
}
cat("\nPer-sample cell counts:\n")
if ("Piece_ID" %in% colnames(combined)) {
  pid_counts <- sort(table(combined$Piece_ID), decreasing = TRUE)
  cat(sprintf("  %d unique samples, range: %d - %d cells/sample\n",
              length(pid_counts), min(pid_counts), max(pid_counts)))
}
