import pytest


def test_core_public_imports():
    # Native GUI/kinematics dependencies are intentionally optional for the
    # lightweight layout check; the full environment.yml enables this smoke test.
    pytest.importorskip("pinocchio")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvista")
    from core import robot_registry, schemas  # noqa: F401
    from collision import collision_manager  # noqa: F401
    from render_engine import render_engine  # noqa: F401
