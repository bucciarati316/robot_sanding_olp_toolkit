from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_demo_urdf_is_present():
    urdf = ROOT / "examples" / "assets" / "urdf" / "demo_six_axis.urdf"
    assert urdf.is_file()
    assert "<robot" in urdf.read_text(encoding="utf-8")


def test_editor_state_is_not_part_of_public_tree():
    assert not (ROOT / ".vscode").exists()
