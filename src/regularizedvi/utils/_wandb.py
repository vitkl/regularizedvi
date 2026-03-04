"""Weights & Biases integration utilities for regularizedvi.

Provides optional W&B experiment tracking via a dual-logger approach
that preserves ``model.history_`` compatibility with scvi-tools'
``SimpleLogger`` while adding live W&B dashboards.

Usage in notebooks::

    from regularizedvi.utils import setup_wandb_logger, log_figure_to_wandb, finish_wandb

    logger, wandb_run = setup_wandb_logger(
        wandb_project="regularizedVI",
        wandb_name="beta5_nobg",
        config={"additive_bg_prior_beta": 5.0},
        results_folder="results/experiment_1/",
    )
    model.train(..., logger=logger)
    fig = model.plot_training_diagnostics()
    log_figure_to_wandb("training_diagnostics", fig)
    finish_wandb()
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure
    import wandb as _wandb_module

logger = logging.getLogger(__name__)


def setup_wandb_logger(
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    wandb_entity: str | None = None,
    wandb_notes: str | None = None,
    wandb_group: str | None = None,
    config: dict | None = None,
    results_folder: str | None = None,
) -> tuple[list | None, _wandb_module.sdk.wandb_run.Run | None]:
    """Set up W&B logger with SimpleLogger for scvi-tools compatibility.

    Returns a dual-logger list ``[SimpleLogger, WandbLogger]`` where
    SimpleLogger is first (so ``trainer.logger`` returns it, preserving
    ``model.history_``), and WandbLogger receives all ``self.log()``
    calls for live experiment tracking.

    Parameters
    ----------
    wandb_project
        W&B project name. If ``None``, returns ``(None, None)`` (no-op).
    wandb_name
        W&B run name (displayed on dashboard).
    wandb_entity
        W&B team/user entity.
    wandb_notes
        Free-text notes for the run.
    wandb_group
        Group name for organizing related runs (e.g., ``"bone_marrow"``).
    config
        Dictionary of hyperparameters to log to W&B.
    results_folder
        Directory for W&B log files.  Creates ``{results_folder}/wandb/``.

    Returns
    -------
    tuple of (logger_list, wandb_run)
        ``logger_list``: ``[SimpleLogger, WandbLogger]`` to pass as
        ``model.train(logger=...)``, or ``None`` if W&B disabled.
        ``wandb_run``: the active ``wandb.Run`` object, or ``None``.
    """
    if wandb_project is None:
        return None, None

    try:
        import wandb
        from lightning.pytorch.loggers import WandbLogger
        from scvi.train._logger import SimpleLogger
    except ImportError as e:
        raise ImportError(
            "wandb and lightning are required for W&B integration. Install with: pip install wandb"
        ) from e

    # Unique run ID from name + timestamp (MD5 hash for safe directory names)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    id_seed = f"{wandb_name or 'run'}_{timestamp}"
    run_id = hashlib.md5(id_seed.encode()).hexdigest()[:8]

    # W&B log directory
    wandb_dir = None
    if results_folder is not None:
        wandb_dir = os.path.join(results_folder, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)

    # Initialize W&B run.
    # console="off" prevents deadlocks when running under papermill:
    # W&B's console capture patches sys.stdout/sys.stderr, which conflicts
    # with papermill's kernel I/O interception (see wandb/wandb#10811).
    wandb_run = wandb.init(
        project=wandb_project,
        name=wandb_name,
        entity=wandb_entity,
        notes=wandb_notes,
        group=wandb_group,
        config=config or {},
        id=run_id,
        dir=wandb_dir,
        reinit=True,
        settings=wandb.Settings(console="off"),
    )

    # Dual-logger: SimpleLogger first (for model.history_ via scvi-tools
    # _safe_load_logger_history which reads trainer.logger.history — Lightning
    # 2.x trainer.logger returns first element of a logger list), WandbLogger
    # second (receives all self.log() calls for live dashboard).
    simple_logger = SimpleLogger()
    wandb_logger = WandbLogger(experiment=wandb_run)
    logger_list = [simple_logger, wandb_logger]

    logger.info("W&B run initialized: %s/%s (id=%s)", wandb_project, wandb_name, run_id)
    return logger_list, wandb_run


def log_figure_to_wandb(name: str, fig: matplotlib.figure.Figure) -> None:
    """Log a matplotlib figure to the active W&B run.

    No-op when no W&B run is active or wandb is not installed.

    Parameters
    ----------
    name
        Key for the logged figure in W&B.
    fig
        Matplotlib figure to log.
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.log({name: wandb.Image(fig)})
    except ImportError:
        pass


def finish_wandb() -> None:
    """Finish the active W&B run.

    No-op when no W&B run is active or wandb is not installed.
    Call at the end of a notebook to ensure all data is uploaded.
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass
