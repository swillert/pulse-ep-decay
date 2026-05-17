"""Smoke test: every name in the public API can be imported."""

from __future__ import annotations


def test_public_api_imports() -> None:
    from pulse_ep_decay import (
        DEFAULT_AMPLITUDE,
        DEFAULT_BASELINE,
        DecayFitResult,
        FocalityResult,
        GeodesicMethod,
        MeshHeatSolver,
        RecoveryResult,
        __version__,
        build_adjacency_matrix,
        compute_focality,
        dijkstra_distances,
        fit_gaussian_decay,
        gaussian_decay,
        generate_synthetic_field,
        heat_distances,
        recover_sigma,
        run_sigma_recovery,
    )

    assert isinstance(__version__, str)
    # Re-bind to silence unused-import warnings; presence is the test.
    _ = (
        DecayFitResult,
        DEFAULT_AMPLITUDE,
        DEFAULT_BASELINE,
        FocalityResult,
        GeodesicMethod,
        MeshHeatSolver,
        RecoveryResult,
        build_adjacency_matrix,
        compute_focality,
        dijkstra_distances,
        fit_gaussian_decay,
        gaussian_decay,
        generate_synthetic_field,
        heat_distances,
        recover_sigma,
        run_sigma_recovery,
    )


def test_integration_extra_guard() -> None:
    """Importing the optional adapter when pulse-ep is *not* installed
    must raise a recognisable ImportError. When pulse-ep IS installed the
    import succeeds — either way the test reflects reality.
    """
    try:
        import pulse_ep_decay.integration.pulse_ep  # noqa: F401
    except ImportError as exc:
        assert "pulse-ep" in str(exc).lower()
