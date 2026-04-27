"""Surgically patch a Jupyter notebook cell.

Supports three operations:
  - replace: overwrite the source of an existing cell (by id)
  - insert:  insert a new cell after the cell with the given id
  - delete:  drop the cell with the given id

The new cell source is read from a file path so large bodies don't
need to be passed on the command line.

Usage examples
--------------
  patch_notebook_cell.py NOTEBOOK --action replace --cell-id params \
      --source-file /tmp/new_params_cell.py

  patch_notebook_cell.py NOTEBOOK --action insert --cell-id params \
      --source-file /tmp/new_cell_body.py --new-cell-id my_new_cell \
      --cell-type code

  patch_notebook_cell.py NOTEBOOK --action delete --cell-id stale_cell

Exit codes
----------
  0 on success; non-zero on failure with a clear error message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat


def main() -> int:
    """Parse CLI args and dispatch to the requested patch operation."""
    p = argparse.ArgumentParser()
    p.add_argument("notebook", type=Path)
    p.add_argument("--action", choices=["replace", "insert", "delete"], required=True)
    p.add_argument("--cell-id", required=True, help="cell id to anchor on")
    p.add_argument("--source-file", type=Path, default=None)
    p.add_argument("--new-cell-id", default=None, help="id for inserted cell (insert only)")
    p.add_argument(
        "--cell-type",
        choices=["code", "markdown"],
        default="code",
        help="cell type for inserted cell (insert only)",
    )
    args = p.parse_args()

    if not args.notebook.is_file():
        print(f"ERROR: notebook not found: {args.notebook}", file=sys.stderr)
        return 2

    if args.action in ("replace", "insert") and args.source_file is None:
        print(f"ERROR: --source-file required for action={args.action}", file=sys.stderr)
        return 2

    nb = nbformat.read(args.notebook, as_version=4)

    idx = next(
        (i for i, c in enumerate(nb.cells) if c.get("id") == args.cell_id),
        None,
    )
    if idx is None:
        print(
            f"ERROR: cell id {args.cell_id!r} not found. Existing ids: {[c.get('id') for c in nb.cells]!r}",
            file=sys.stderr,
        )
        return 3

    if args.action == "replace":
        new_src = args.source_file.read_text()
        nb.cells[idx]["source"] = new_src
        nb.cells[idx]["outputs"] = []
        nb.cells[idx]["execution_count"] = None
        if args.new_cell_id is not None:
            nb.cells[idx]["id"] = args.new_cell_id
            print(f"Replaced cell {idx} (id={args.cell_id} -> {args.new_cell_id})")
        else:
            print(f"Replaced cell {idx} (id={args.cell_id})")

    elif args.action == "insert":
        new_src = args.source_file.read_text()
        if args.cell_type == "code":
            new_cell = nbformat.v4.new_code_cell(source=new_src)
        else:
            new_cell = nbformat.v4.new_markdown_cell(source=new_src)
        if args.new_cell_id is not None:
            new_cell["id"] = args.new_cell_id
        nb.cells.insert(idx + 1, new_cell)
        print(f"Inserted new cell at index {idx + 1} (after id={args.cell_id})")

    elif args.action == "delete":
        del nb.cells[idx]
        print(f"Deleted cell {idx} (id={args.cell_id})")

    nbformat.write(nb, args.notebook)
    return 0


if __name__ == "__main__":
    sys.exit(main())
