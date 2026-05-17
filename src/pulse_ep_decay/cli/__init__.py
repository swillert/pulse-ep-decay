"""Console-script entry points for ``pulse-ep-decay``.

Each module exposes a ``main(argv=None) -> int`` function so the same
code is callable from the shell (via the installed ``pulse-ep-decay-*``
binaries declared in ``pyproject.toml``) and from Python::

    from pulse_ep_decay.cli import spatial_decay
    spatial_decay.main(["--mesh", "atrium.npz", "--out", "decay.csv"])
"""
