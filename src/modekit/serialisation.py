"""Serialisation of ``equinox.Module`` models.

A single file stores, on its first line, a JSON object of *hyperparameters* --
everything a builder needs to reconstruct the model's skeleton (its pytree
structure, static fields, and array shapes/dtypes) -- and, after a newline, the
raw array leaves written by :func:`equinox.tree_serialise_leaves`.

This is the pattern from the Equinox documentation, generalised so the same two
functions serve any model (``EmaModel``, ``OmaModel``, a ``ModalModel`` later;
batched or not): the caller supplies a ``make(**hyperparams) -> skeleton``
builder. One twist versus the docs, whose ``make`` allocates fresh arrays from
scalar sizes: our constructors do heavy FFT work at ``__init__``, so ``make``
should build the skeleton with :func:`equinox.filter_eval_shape` (shapes only,
no compute) -- which is why the array shapes/dtypes travel inside
``hyperparams``.
"""

import hashlib
import json
from pathlib import Path

import equinox as eqx


def save_model(path, model, hyperparams):
    """Serialise ``model`` to ``path``: a JSON ``hyperparams`` header + eqx leaves.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file (conventionally ``*.eqx``). Parent directories are
        created if missing.
    model : eqx.Module
        The model, or a batched stack of models, to serialise.
    hyperparams : dict
        JSON-serialisable data sufficient for a matching ``make`` to rebuild the
        skeleton at load time: the static fields plus the array shapes/dtypes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write((json.dumps(hyperparams) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(path, make):
    """Inverse of :func:`save_model`.

    Reads the ``hyperparams`` header, rebuilds a skeleton with
    ``make(**hyperparams)``, then reads the array leaves back into it.

    Parameters
    ----------
    path : str or pathlib.Path
        A file written by :func:`save_model`.
    make : callable
        ``make(**hyperparams) -> skeleton``: a pytree with the same structure as
        the saved model, the correct static fields, and matching leaf
        shapes/dtypes. Typically ``eqx.filter_eval_shape(constructor,
        *abstract_inputs)`` so no computation runs.

    Returns
    -------
    eqx.Module
        The deserialised model.
    """
    with open(path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        skeleton = make(**hyperparams)
        return eqx.tree_deserialise_leaves(f, skeleton)


def cache_key(*parts, n=10):
    """Short, stable hex digest of arbitrary JSON-able ``parts``.

    Build cache filenames whose identity tracks whatever material is passed
    (settings, file fingerprints, ...); any change to the material flips the
    key. The caller decides *what* goes in -- this only hashes it.

    Parameters
    ----------
    *parts
        JSON-serialisable values. Anything unserialisable falls back to ``str``.
    n : int, optional
        Number of leading hex characters to keep. Default 10.

    Returns
    -------
    str
        The first ``n`` hex characters of the md5 digest.
    """
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:n]
