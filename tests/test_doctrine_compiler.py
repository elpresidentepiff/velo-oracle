from pathlib import Path

from mesh.neosym import DoctrineCompiler


def test_doctrine_compiler_loads_default_yaml(tmp_path: Path) -> None:
    source = Path("mesh/neosym/doctrine_lang.yaml")
    destination = tmp_path / "doctrine_lang.yaml"
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    compiler = DoctrineCompiler(destination)
    graphs = compiler.load()

    assert "doctrine_A_form_cycle" in graphs
    doctrine = graphs["doctrine_A_form_cycle"]
    assert doctrine.rules
    assert doctrine.rules[0].conditions
