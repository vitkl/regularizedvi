"""Neural network building blocks for regularizedvi.

Modified from scvi-tools (scvi.nn._base_components) with the following changes:

RegularizedFCLayers (from FCLayers):
- New ``dropout_on_input`` parameter: applies dropout BEFORE the linear layer
- Default ``use_batch_norm=False``, ``use_layer_norm=True`` (LayerNorm preferred)

RegularizedEncoder (from Encoder):
- Uses ``RegularizedFCLayers`` instead of ``FCLayers``
- New ``"softplus"`` distribution option for non-negative latent transformations

RegularizedDecoderSCVI (from DecoderSCVI):
- Uses ``RegularizedFCLayers`` instead of ``FCLayers``
- New ``additive_background`` parameter in ``forward()`` for ambient RNA correction
- Rate computation: ``library_act(library) * (px_scale + additive_background)``
"""

import collections
from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch import nn
from torch.distributions import Normal


def _identity(x):
    return x


class RegularizedFCLayers(nn.Module):
    """Fully-connected layers with optional dropout-on-input.

    When ``dropout_on_input=True``, dropout is applied BEFORE the linear layer
    (feature-level masking) rather than after activation. This prevents over-reliance
    on any single gene and works better with the regularizedvi model.

    Default normalization is LayerNorm (not BatchNorm), as LayerNorm normalises
    across features within each sample, making it independent of batch composition.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    dropout_on_input
        If True, apply dropout before the linear layer instead of after activation.
        Layer order becomes: Dropout -> Linear -> BatchNorm -> LayerNorm -> Activation.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
        dropout_on_input: bool = False,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        self.dropout_on_input = dropout_on_input
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)

        if dropout_on_input:
            # Dropout -> Linear -> BatchNorm -> LayerNorm -> Activation (no trailing dropout)
            self.fc_layers = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            f"Layer {i}",
                            nn.Sequential(
                                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                                nn.Linear(
                                    n_in + cat_dim * self.inject_into_layer(i),
                                    n_out,
                                    bias=bias,
                                ),
                                nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                                nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                                activation_fn() if use_activation else None,
                            ),
                        )
                        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:], strict=True))
                    ]
                )
            )
        else:
            # Standard order: Linear -> BatchNorm -> LayerNorm -> Activation -> Dropout
            self.fc_layers = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            f"Layer {i}",
                            nn.Sequential(
                                nn.Linear(
                                    n_in + cat_dim * self.inject_into_layer(i),
                                    n_out,
                                    bias=bias,
                                ),
                                nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                                nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                                activation_fn() if use_activation else None,
                                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                            ),
                        )
                        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:], strict=True))
                    ]
                )
            )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1))) for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


class RegularizedEncoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses ``RegularizedFCLayers`` with LayerNorm by default and optional dropout-on-input.
    Adds ``"softplus"`` distribution option for non-negative latent transformations,
    needed for the ambient RNA additive background coefficient.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z. One of ``"normal"``, ``"ln"`` (logistic normal),
        or ``"softplus"`` (non-negative via softplus transformation).
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :meth:`torch.exp`.
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`RegularizedFCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        return_dist: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = RegularizedFCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        elif distribution == "softplus":
            self.z_transformation = nn.Softplus()
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal
            \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent


class RegularizedDecoderSCVI(nn.Module):
    """Decoder with ambient RNA additive background support.

    Uses ``RegularizedFCLayers`` with LayerNorm by default.
    The ``forward()`` method accepts an optional ``additive_background`` tensor
    that is added to ``px_scale`` before multiplying by library size, implementing
    the ambient RNA correction: ``rate = library_act(library) * (px_scale + background)``.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    **kwargs
        Keyword args for :class:`RegularizedFCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        decoder_type: str = "expected_RNA",
        burst_size_n_hidden: int | None = None,
        burst_size_intercept: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.decoder_type = decoder_type
        self.burst_size_intercept = burst_size_intercept
        self.px_decoder = RegularizedFCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        # Secondary decoder for burst_frequency_size (burst_size head)
        if decoder_type == "burst_frequency_size":
            _bs_n_hidden = burst_size_n_hidden if burst_size_n_hidden is not None else max(1, n_hidden // 2)
            self.burst_size_decoder = RegularizedFCLayers(
                n_in=n_input,
                n_out=_bs_n_hidden,
                n_cat_list=n_cat_list,
                n_layers=max(1, n_layers - 1),
                n_hidden=_bs_n_hidden,
                dropout_rate=0,
                inject_covariates=inject_covariates,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                **kwargs,
            )
            self.burst_size_head = nn.Sequential(
                nn.Linear(_bs_n_hidden, n_output),
                nn.Softplus(),
            )

        # px_r_decoder kept for backward compatibility with saved models (gene-cell deprecated)
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
        additive_background: torch.Tensor | None = None,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library
            library size (log-scale)
        cat_list
            list of category membership(s) for this sample
        additive_background
            Per-gene, per-cell additive background (ambient RNA).
            Shape ``(n_cells, n_genes)``. Added to ``px_scale`` before
            multiplying by library size.

        Returns
        -------
        5-tuple or 7-tuple of :py:class:`torch.Tensor`
            ``(px_scale, None, px_rate, px_dropout, px)`` for expected_RNA, or
            ``(px_scale, None, px_rate, px_dropout, px, burst_freq, burst_size)``
            for burst_frequency_size. ``px`` is the hidden activations from the
            decoder FC layers (before the scale head).

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)

        if self.decoder_type == "burst_frequency_size":
            # px_scale is burst_frequency (softplus output from main decoder)
            burst_freq = px_scale
            burst_size = self.burst_size_head(self.burst_size_decoder(z, *cat_list)) + self.burst_size_intercept

            # Ambient RNA: add per-gene, per-batch background before scaling by library
            if additive_background is not None:
                px_rate = torch.exp(library) * (burst_freq * burst_size + additive_background)
            else:
                px_rate = torch.exp(library) * burst_freq * burst_size

            return px_scale, None, px_rate, px_dropout, px, burst_freq, burst_size
        else:
            # Standard expected_RNA path
            if additive_background is not None:
                px_rate = torch.exp(library) * (px_scale + additive_background)
            else:
                px_rate = torch.exp(library) * px_scale

            return px_scale, None, px_rate, px_dropout, px
