r"""Generate a multi-page HTML report from experiment output notebooks.

Reads experiments.tsv, extracts images from key cells in output notebooks,
and produces:
  - experiment_report.html          — lightweight index (table of contents)
  - experiment_pages/*.html         — one page per experiment with param table
  - experiment_pages/*.png          — extracted plot images (not base64-embedded)

Configuration
-------------
CELL_PATTERNS (dict):
    Maps display names to cell source search patterns. Each code cell in a
    notebook is scanned; if the pattern substring appears in the cell source,
    all PNG outputs from that cell are extracted.

    To adapt for different notebooks or datasets, edit CELL_PATTERNS below.

    Example for spatial transcriptomics:
        CELL_PATTERNS = {
            "Spatial Plot": "plot_spatial",
            "Deconvolution": "plot_deconvolution",
        }

    Example for immune integration:
        CELL_PATTERNS = {
            "QC Metrics": "plot_qc",
            "Harmony UMAP": "sc.pl.embedding",
            "Marker Dotplot": "sc.pl.dotplot",
        }

REPO_ROOT / BASE_DIR:
    Change these to point at a different repository or notebook directory.
    TSV_PATH is derived from BASE_DIR.

experiments.tsv schema:
    Required columns: name, type, era, notebook, status, label
    All other columns are treated as hyperparameters and shown in the
    per-experiment param table when non-empty. The 'notebook' column must
    be a repo-relative path to the output .ipynb file.

Usage:
    bash scripts/helper_scripts/run_python_cmd.sh \\
        scripts/claude_helper_scripts/generate_experiment_report.py
"""

import base64
import csv
import glob
import html
import json
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths — change these for a different project / dataset
# ---------------------------------------------------------------------------
REPO_ROOT = "/nfs/team205/vk7/sanger_projects/my_packages/regularizedvi"
BASE_DIR = os.path.join(REPO_ROOT, "docs/notebooks/model_comparisons")
TSV_PATH = os.path.join(BASE_DIR, "experiments.tsv")
OUTPUT_INDEX = os.path.join(BASE_DIR, "experiment_report.html")
PAGES_DIR = os.path.join(BASE_DIR, "experiment_pages")

# ---------------------------------------------------------------------------
# Cell selection — edit this dict to extract different plots
# ---------------------------------------------------------------------------
# Keys   = display name shown in HTML
# Values = substring searched in each code cell's source text
#
# A cell matches if  `pattern in cell_source`  is True.
# All PNG images from matching cells are extracted.
# Order here determines display order on experiment pages.
CELL_PATTERNS = {
    "Training Diagnostics": "plot_training_diagnostics",
    "UMAP Comparison": "plot_umap_comparison",
    "Attribution Scatter": "plot_attribution_scatter",
    "Modality Attribution": "plot_modality_attribution",
}

# Metadata columns skipped in the hyperparameter table
_SKIP_COLS = {"name", "type", "era", "results_folder", "label", "notebook", "status"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cell_source(cell):
    """Return the full source text of a notebook cell."""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def _extract_images_from_cell(cell):
    """Return list of raw PNG bytes from a cell's outputs."""
    images = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        img = data.get("image/png")
        if img is not None:
            if isinstance(img, list):
                img = "".join(img)
            img = img.strip()
            images.append(base64.b64decode(img))
    return images


def _extract_plots_from_notebook(nb_path):
    """Extract images keyed by display name from a notebook file.

    Returns dict: {display_name: [raw_png_bytes, ...]}
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    results = {name: [] for name in CELL_PATTERNS}

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = _cell_source(cell)
        for display_name, pattern in CELL_PATTERNS.items():
            if pattern in src:
                images = _extract_images_from_cell(cell)
                results[display_name].extend(images)

    return results


def _save_images(experiment_name, plots):
    """Write extracted PNG images to disk.

    Returns (paths_dict, n_images) where paths_dict maps
    display_name -> [filename, ...] (filenames only, no directory prefix).
    """
    paths = {name: [] for name in plots}
    img_counter = 0
    for display_name in CELL_PATTERNS:
        for png_bytes in plots.get(display_name, []):
            filename = f"{experiment_name}_img_{img_counter}.png"
            filepath = os.path.join(PAGES_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(png_bytes)
            paths[display_name].append(filename)
            img_counter += 1
    return paths, img_counter


def _build_param_table(row, headers):
    """Build an HTML table of non-empty hyperparameters."""
    rows_html = []
    for h in headers:
        if h in _SKIP_COLS:
            continue
        val = (row.get(h) or "").strip()
        if val:
            rows_html.append(f"      <tr><td>{html.escape(h)}</td><td>{html.escape(val)}</td></tr>")
    if not rows_html:
        return "<p><em>No hyperparameters set (all defaults).</em></p>"
    return (
        '    <table class="params">\n'
        "      <tr><th>Parameter</th><th>Value</th></tr>\n" + "\n".join(rows_html) + "\n    </table>"
    )


def _page_style():
    """Return shared CSS <style> block for both index and experiment pages."""
    return """<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
    background: #fafafa;
    color: #333;
  }
  h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
  h2 {
    margin-top: 40px;
    padding: 10px 15px;
    background: #e8e8e8;
    border-left: 4px solid #4a90d9;
  }
  h3 { color: #4a90d9; margin-top: 20px; }
  a { color: #4a90d9; text-decoration: none; }
  a:hover { text-decoration: underline; }
  table.params {
    border-collapse: collapse;
    margin: 10px 0;
  }
  table.params th, table.params td {
    border: 1px solid #ccc;
    padding: 4px 10px;
    text-align: left;
  }
  table.params th { background: #f0f0f0; }
  table.index {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
  }
  table.index th, table.index td {
    border: 1px solid #ddd;
    padding: 6px 12px;
    text-align: left;
  }
  table.index th { background: #f0f0f0; }
  table.index tr:hover { background: #f5f5f5; }
  .nb-path { font-size: 0.85em; color: #777; margin-top: -10px; }
  .nb-path code { background: #eee; padding: 2px 5px; border-radius: 3px; }
  .exp-status {
    font-size: 0.85em;
    padding: 2px 8px;
    border-radius: 3px;
    margin-left: 8px;
  }
  .status-completed { background: #d4edda; color: #155724; }
  .status-pending { background: #fff3cd; color: #856404; }
  .status-failed { background: #f8d7da; color: #721c24; }
  .type-tag {
    font-size: 0.8em;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
  }
  .type-rna { background: #cfe2f3; color: #1a5276; }
  .type-multimodal { background: #d5a6bd; color: #4a1942; }
  .plot-section { margin: 15px 0; }
  .plot-section img {
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
    border-radius: 4px;
    margin: 5px 0;
    background: #fff;
  }
  .missing { color: #999; font-style: italic; }
  .era-row td {
    background: #4a90d9 !important;
    color: #fff;
    font-weight: 600;
    font-size: 1.1em;
  }
  .generated { font-size: 0.85em; color: #999; }
  .back-link { margin-bottom: 20px; }
</style>"""


# ---------------------------------------------------------------------------
# Page generators
# ---------------------------------------------------------------------------


def _generate_experiment_page(exp, headers, image_filenames, page_filename):
    """Generate a single experiment HTML page.

    Args:
        exp: dict row from experiments.tsv
        headers: list of column names
        image_filenames: dict {display_name: [filename, ...]} (files in same dir)
        page_filename: filename of this page (for reference)

    Returns
    -------
        n_images written to page
    """
    name = exp.get("name", "unknown")
    label = exp.get("label", name)
    nb_file = exp.get("notebook", "").strip()
    status = exp.get("status", "")
    exp_type = exp.get("type", "")
    status_cls = f"status-{status}" if status in ("completed", "pending", "failed") else ""

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(label)}</title>
{_page_style()}
</head>
<body>
<p class="back-link"><a href="../experiment_report.html">&larr; Back to index</a></p>
<h1>{html.escape(label)}
  <span class="exp-status {status_cls}">{html.escape(status)}</span>
  <span class="type-tag type-{html.escape(exp_type)}">{html.escape(exp_type)}</span>
</h1>
""")

    if nb_file:
        parts.append(f'<p class="nb-path">Notebook: <code>{html.escape(nb_file)}</code></p>\n')

    # Parameter table
    parts.append(_build_param_table(exp, headers))
    parts.append("\n")

    # Images by category
    total_imgs = 0
    for display_name in CELL_PATTERNS:
        img_list = image_filenames.get(display_name, [])
        if not img_list:
            continue
        parts.append('<div class="plot-section">\n')
        parts.append(f"  <h3>{html.escape(display_name)}</h3>\n")
        for img_fn in img_list:
            parts.append(f'  <img src="{html.escape(img_fn)}" alt="{html.escape(display_name)}" loading="lazy">\n')
            total_imgs += 1
        parts.append("</div>\n")

    if total_imgs == 0:
        parts.append('<p class="missing">No target plots found in this notebook.</p>\n')

    parts.append("</body>\n</html>\n")

    page_path = os.path.join(PAGES_DIR, page_filename)
    with open(page_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    return total_imgs


def _generate_index(experiments, page_info):
    """Generate the lightweight index HTML page (table of contents)."""
    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment Report — Index</title>
{_page_style()}
</head>
<body>
<h1>Experiment Report</h1>
<p class="generated">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
""")

    # Era-grouped table
    parts.append('<table class="index">\n')
    parts.append("  <tr><th>Experiment</th><th>Notebook</th><th>Type</th><th>Status</th><th>Images</th></tr>\n")

    current_era = None
    for exp, page_filename, n_imgs in page_info:
        era = exp.get("era", "")
        if era != current_era:
            parts.append(f'  <tr class="era-row"><td colspan="5">Era {html.escape(era)}</td></tr>\n')
            current_era = era

        name = exp.get("name", "unknown")
        label = exp.get("label", name)
        nb_file = exp.get("notebook", "").strip()
        status = exp.get("status", "")
        exp_type = exp.get("type", "")
        status_cls = f"status-{status}" if status in ("completed", "pending", "failed") else ""

        link = f"experiment_pages/{page_filename}"
        parts.append(
            f"  <tr>"
            f'<td><a href="{html.escape(link)}">{html.escape(label)}</a></td>'
            f"<td><code>{html.escape(nb_file)}</code></td>"
            f'<td><span class="type-tag type-{html.escape(exp_type)}">'
            f"{html.escape(exp_type)}</span></td>"
            f'<td><span class="exp-status {status_cls}">'
            f"{html.escape(status)}</span></td>"
            f"<td>{n_imgs}</td>"
            f"</tr>\n"
        )

    parts.append("</table>\n")
    parts.append("</body>\n</html>\n")

    with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
        f.write("".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_report():
    """Generate the multi-page experiment report."""
    # Read TSV
    with open(TSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames
        experiments = list(reader)

    print(f"Loaded {len(experiments)} experiments from {TSV_PATH}")

    # Create pages directory and clean stale files
    os.makedirs(PAGES_DIR, exist_ok=True)
    for old_file in glob.glob(os.path.join(PAGES_DIR, "*.png")):
        os.remove(old_file)
    for old_file in glob.glob(os.path.join(PAGES_DIR, "*.html")):
        os.remove(old_file)
    print(f"Cleaned {PAGES_DIR}/")

    # Pass 1: generate per-experiment pages
    page_info = []  # list of (exp, page_filename, n_images)
    for exp in experiments:
        name = exp.get("name", "unknown").strip()
        if not name:
            continue
        nb_file = exp.get("notebook", "").strip()
        page_filename = f"{name}.html"

        if not nb_file:
            print(f"  [{name}] No notebook — skeleton page")
            n_imgs = _generate_experiment_page(exp, headers, {}, page_filename)
            page_info.append((exp, page_filename, n_imgs))
            continue

        nb_path = os.path.join(REPO_ROOT, nb_file)
        if not os.path.isfile(nb_path):
            print(f"  [{name}] Not found: {nb_file} — skeleton page")
            n_imgs = _generate_experiment_page(exp, headers, {}, page_filename)
            page_info.append((exp, page_filename, n_imgs))
            continue

        print(f"  [{name}] Processing ...", end="", flush=True)
        try:
            plots = _extract_plots_from_notebook(nb_path)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f" ERROR: {e}")
            _generate_experiment_page(exp, headers, {}, page_filename)
            page_info.append((exp, page_filename, 0))
            continue

        image_filenames, n_imgs = _save_images(name, plots)
        _generate_experiment_page(exp, headers, image_filenames, page_filename)
        page_info.append((exp, page_filename, n_imgs))
        print(f" {n_imgs} images")

    # Pass 2: generate index
    _generate_index(experiments, page_info)

    total_images = sum(n for _, _, n in page_info)
    print(f"\nDone: {len(page_info)} pages, {total_images} images total")
    print(f"Index: {OUTPUT_INDEX}")
    print(f"Pages: {PAGES_DIR}/")


if __name__ == "__main__":
    generate_report()
