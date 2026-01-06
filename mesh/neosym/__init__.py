"""NeuroSymbolic Core package."""

from pathlib import Path
from typing import Mapping

from .compiler import DoctrineCompiler, DoctrineGraph
from .executor import DoctrineEvaluation, DoctrineExecutor

__all__ = [
    "DoctrineCompiler",
    "DoctrineExecutor",
    "DoctrineGraph",
    "DoctrineEvaluation",
]


def load_default_doctrines(base_path: Path) -> Mapping[str, DoctrineGraph]:
    """Load doctrine definitions relative to the repository root."""

    compiler = DoctrineCompiler(base_path / "mesh" / "neosym" / "doctrine_lang.yaml")
    return compiler.load()
