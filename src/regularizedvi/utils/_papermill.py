"""Type coercion for papermill ``-r`` parameters.

Papermill's ``-r`` flag always injects values as strings into notebooks,
which breaks ``bool("0") == True`` and ``float_val / "5.0"``.

This utility converts string parameters to their intended Python types.
"""

from __future__ import annotations


def coerce_papermill_params(
    **params: tuple[object, type | str],
) -> dict[str, object]:
    """Convert papermill ``-r`` string params to correct Python types.

    Each keyword argument is a ``(value, target_type)`` tuple where
    ``target_type`` is one of:

    - ``bool`` — ``"0"``/``"1"`` or ints → ``False``/``True``
      (via ``int()`` first to avoid ``bool("0") == True``)
    - ``float`` — ``"5.0"`` → ``5.0``
    - ``int`` — ``"42"`` → ``42``
    - ``"str_or_none"`` — ``"None"``/``"none"`` → ``None``,
      ``None`` → ``None``, other strings pass through

    Values that are already the target type (or ``None``) pass through
    unchanged.

    Parameters
    ----------
    **params
        ``name=(value, target_type)`` pairs.

    Returns
    -------
    dict
        ``{name: coerced_value}`` for each input parameter.

    Examples
    --------
    >>> coerce_papermill_params(
    ...     regularise_background=("0", bool),
    ...     additive_bg_prior_beta=("5.0", float),
    ...     wandb_project=("None", "str_or_none"),
    ... )
    {'regularise_background': False, 'additive_bg_prior_beta': 5.0, 'wandb_project': None}
    """
    result = {}
    for name, (value, target_type) in params.items():
        result[name] = _coerce_single(value, target_type, name)
    return result


def _coerce_single(value: object, target_type: type | str, name: str) -> object:
    """Coerce a single value to the target type."""
    # None passthrough
    if value is None:
        return None

    # str_or_none: "None"/"none" → None, None → None, else str
    if target_type == "str_or_none":
        if isinstance(value, str) and value.lower() == "none":
            return None
        return value

    # float_or_none: "None"/"none" → None, None → None, else float
    if target_type == "float_or_none":
        if isinstance(value, str) and value.lower() == "none":
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            msg = f"Cannot coerce {name}={value!r} to float_or_none"
            raise TypeError(msg) from e

    # Already correct type — passthrough
    if isinstance(value, target_type) and not (target_type is float and isinstance(value, bool)):
        return value

    # bool: must go through int() first to handle "0" → False correctly
    if target_type is bool:
        try:
            return bool(int(value))
        except (ValueError, TypeError) as e:
            msg = f"Cannot coerce {name}={value!r} to bool"
            raise TypeError(msg) from e

    # float / int
    if target_type in (float, int):
        try:
            return target_type(value)
        except (ValueError, TypeError) as e:
            msg = f"Cannot coerce {name}={value!r} to {target_type.__name__}"
            raise TypeError(msg) from e

    msg = f"Unsupported target_type={target_type!r} for {name}"
    raise TypeError(msg)
