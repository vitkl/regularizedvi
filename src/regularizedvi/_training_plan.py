"""Custom training plan with per-modality learning rate support."""

from __future__ import annotations

from scvi.train import TrainingPlan
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MultimodalTrainingPlan(TrainingPlan):
    """Training plan that supports per-modality learning rate multipliers.

    When ``modality_lr_multiplier`` is ``None``, behaves identically to the
    default scvi-tools ``TrainingPlan``.  When set, creates separate optimizer
    parameter groups for each modality with scaled learning rates.

    Parameters
    ----------
    module
        The VAE module (must implement ``get_parameter_groups`` when
        ``modality_lr_multiplier`` is not ``None``).
    modality_lr_multiplier
        Mapping from modality name to LR multiplier, e.g. ``{"atac": 2.0}``.
        Modalities not listed use the base learning rate.
    **kwargs
        Forwarded to ``TrainingPlan.__init__``.
    """

    def __init__(
        self,
        module,
        *,
        modality_lr_multiplier: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(module, **kwargs)
        self.modality_lr_multiplier = modality_lr_multiplier

    def configure_optimizers(self):
        """Configure optimizer with optional per-modality param groups."""
        if self.modality_lr_multiplier is None:
            return super().configure_optimizers()

        param_groups = self.module.get_parameter_groups(
            base_lr=self.lr,
            modality_lr_multiplier=self.modality_lr_multiplier,
        )

        optimizer = self.get_optimizer_creator()(param_groups)

        config = {"optimizer": optimizer}
        if self.reduce_lr_on_plateau:
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )
        return config
