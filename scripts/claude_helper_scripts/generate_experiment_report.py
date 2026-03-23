"""Generate an HTML report from experiment output notebooks.

Reads experiments.tsv, extracts training diagnostics, UMAP comparison,
and attribution images from output notebooks, and produces a single
self-contained HTML report.

Usage:
    bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/generate_experiment_report.py
"""

import csv
import html
import json
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = "/nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/docs/notebooks/model_comparisons"
TSV_PATH = os.path.join(BASE_DIR, "experiments.tsv")
OUTPUT_HTML = os.path.join(BASE_DIR, "experiment_report.html")

# Function names to search for in cell sources
TARGET_FUNCTIONS = [
    "plot_training_diagnostics",
    "plot_umap_comparison",
    "plot_attribution_scatter",
    "plot_modality_attribution",
]


def _cell_source(cell):
    """Return the full source text of a notebook cell."""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def _extract_images_from_cell(cell):
    """Return list of base64 PNG strings from a cell's outputs."""
    images = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        img = data.get("image/png")
        if img is not None:
            # image/png can be a string or a list of strings
            if isinstance(img, list):
                img = "".join(img)
            # Strip whitespace/newlines that may be present
            img = img.strip()
            images.append(img)
    return images


def _extract_plots_from_notebook(nb_path):
    """Extract images keyed by function name from a notebook file.

    Returns a dict: {function_name: [base64_png, ...]}
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    results = {fn: [] for fn in TARGET_FUNCTIONS}

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = _cell_source(cell)
        for fn in TARGET_FUNCTIONS:
            if fn in src:
                images = _extract_images_from_cell(cell)
                results[fn].extend(images)

    return results


def _build_param_table(row, headers):
    """Build an HTML table of non-empty hyperparameters.

    Skips the metadata columns (name, type, era, results_folder, label,
    notebook, status) and only shows hyperparameter columns that have values.
    """
    skip_cols = {"name", "type", "era", "results_folder", "label", "notebook", "status"}
    rows_html = []
    for h in headers:
        if h in skip_cols:
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


def _section_label(fn):
    """Human-readable label for a target function name."""
    return {
        "plot_training_diagnostics": "Training Diagnostics",
        "plot_umap_comparison": "UMAP Comparison",
        "plot_attribution_scatter": "Attribution Scatter",
        "plot_modality_attribution": "Modality Attribution",
    }.get(fn, fn)


def generate_report():  # noqa: D103
    # Read TSV
    with open(TSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames
        experiments = list(reader)

    print(f"Loaded {len(experiments)} experiments from {TSV_PATH}")

    # Start building HTML
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment Report</title>
<style>
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
  .toc {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 15px 25px;
    margin-bottom: 30px;
  }
  .toc ul { columns: 2; -webkit-columns: 2; list-style: none; padding-left: 0; }
  .toc li { margin-bottom: 4px; }
  .toc a { text-decoration: none; color: #4a90d9; }
  .toc a:hover { text-decoration: underline; }
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
  .nb-path { font-size: 0.85em; color: #777; margin-top: -10px; }
  .exp-status { font-size: 0.85em; padding: 2px 8px; border-radius: 3px; margin-left: 10px; }
  .status-completed { background: #d4edda; color: #155724; }
  .status-pending { background: #fff3cd; color: #856404; }
  .status-failed { background: #f8d7da; color: #721c24; }
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
  .era-divider {
    margin-top: 50px;
    padding: 8px 15px;
    background: #4a90d9;
    color: #fff;
    font-size: 1.3em;
    border-radius: 4px;
  }
</style>
</head>
<body>
<h1>Experiment Report</h1>
""")

    # Table of contents
    html_parts.append('<div class="toc">\n<h3>Table of Contents</h3>\n<ul>\n')
    current_era = None
    for exp in experiments:
        era = exp.get("era", "")
        if era != current_era:
            if current_era is not None:
                html_parts.append("</ul>\n<ul>\n")
            html_parts.append(f"  <li><strong>Era {html.escape(era)}</strong></li>\n")
            current_era = era
        name = exp.get("name", "unknown")
        label = exp.get("label", name)
        status = exp.get("status", "")
        status_cls = f"status-{status}" if status in ("completed", "pending", "failed") else ""
        html_parts.append(
            f'  <li><a href="#{html.escape(name)}">{html.escape(label)}</a>'
            f' <span class="exp-status {status_cls}">{html.escape(status)}</span></li>\n'
        )
    html_parts.append("</ul>\n</div>\n")

    # Experiment sections
    current_era = None
    for exp in experiments:
        name = exp.get("name", "unknown")
        label = exp.get("label", name)
        nb_file = exp.get("notebook", "").strip()
        status = exp.get("status", "")
        era = exp.get("era", "")
        exp_type = exp.get("type", "")

        if era != current_era:
            html_parts.append(f'<div class="era-divider">Era {html.escape(era)}</div>\n')
            current_era = era

        status_cls = f"status-{status}" if status in ("completed", "pending", "failed") else ""
        html_parts.append(
            f'<h2 id="{html.escape(name)}">{html.escape(label)}'
            f' <span class="exp-status {status_cls}">{html.escape(status)}</span>'
            f" [{html.escape(exp_type)}]</h2>\n"
        )

        if nb_file:
            html_parts.append(f'<p class="nb-path">Notebook: {html.escape(nb_file)}</p>\n')

        # Parameter table
        html_parts.append(_build_param_table(exp, headers))
        html_parts.append("\n")

        # Try to load notebook
        if not nb_file:
            html_parts.append('<p class="missing">No notebook specified.</p>\n')
            print(f"  [{name}] No notebook specified — skipping")
            continue

        nb_path = os.path.join(BASE_DIR, nb_file)
        if not os.path.isfile(nb_path):
            html_parts.append(f'<p class="missing">Notebook not found: {html.escape(nb_file)}</p>\n')
            print(f"  [{name}] Notebook not found: {nb_file} — skipping")
            continue

        print(f"  [{name}] Processing {nb_file} ...", end="")

        try:
            plots = _extract_plots_from_notebook(nb_path)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            html_parts.append(f'<p class="missing">Error reading notebook: {html.escape(str(e))}</p>\n')
            print(f" ERROR: {e}")
            continue

        n_images = 0
        for fn in TARGET_FUNCTIONS:
            images = plots[fn]
            if not images:
                continue
            html_parts.append('<div class="plot-section">\n')
            html_parts.append(f"  <h3>{_section_label(fn)}</h3>\n")
            for img_b64 in images:
                html_parts.append(
                    f'  <img src="data:image/png;base64,{img_b64}" alt="{html.escape(_section_label(fn))}">\n'
                )
                n_images += 1
            html_parts.append("</div>\n")

        if n_images == 0:
            html_parts.append('<p class="missing">No target plots found in this notebook.</p>\n')
        print(f" {n_images} image(s) extracted")

    html_parts.append("</body>\n</html>\n")

    # Write output
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"\nReport written to {OUTPUT_HTML}")


if __name__ == "__main__":
    generate_report()
