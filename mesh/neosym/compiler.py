"""Compiler for declarative doctrine definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback parsing path
    yaml = None


@dataclass(slots=True)
class DoctrineCondition:
    """Single condition expression bound to a rule."""

    expression: str


@dataclass(slots=True)
class DoctrineRule:
    """Executable doctrine rule containing one or more conditions."""

    name: str
    conditions: Sequence[DoctrineCondition]
    score: float
    action: str


@dataclass(slots=True)
class DoctrineGraph:
    """Collection of rules for a single doctrine."""

    doctrine_id: str
    rules: Sequence[DoctrineRule] = field(default_factory=tuple)


class DoctrineCompiler:
    """Compiles YAML doctrine definitions into executable graphs."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> Mapping[str, DoctrineGraph]:
        """Load doctrine definitions from disk."""

        with self._path.open("r", encoding="utf-8") as handle:
            text = handle.read()
        raw = _load_doctrine_text(text)
        return self._compile(raw)

    def _compile(self, raw: Mapping[str, Mapping[str, Mapping[str, object]]]) -> Mapping[str, DoctrineGraph]:
        graphs: Dict[str, DoctrineGraph] = {}
        for doctrine_id, rules in raw.items():
            compiled_rules: Iterable[DoctrineRule] = (
                DoctrineRule(
                    name=rule_name,
                    conditions=tuple(
                        DoctrineCondition(expression=str(condition))
                        for condition in rule_config.get("conditions", [])
                    ),
                    score=float(rule_config.get("score", 0.0)),
                    action=str(rule_config.get("action", "")),
                )
                for rule_name, rule_config in rules.items()
            )
            graphs[doctrine_id] = DoctrineGraph(doctrine_id=doctrine_id, rules=tuple(compiled_rules))
        return graphs


def _load_doctrine_text(
    text: str,
) -> MutableMapping[str, MutableMapping[str, MutableMapping[str, object]]]:
    """Load doctrine definitions using PyYAML if available, otherwise fallback."""

    if yaml is not None:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, Mapping):
            raise ValueError("Doctrine definition must be a mapping")
        return data
    return _parse_doctrine_fallback(text)


def _parse_doctrine_fallback(
    text: str,
) -> MutableMapping[str, MutableMapping[str, MutableMapping[str, object]]]:
    """Parse a constrained YAML subset for environments without PyYAML."""

    doctrines: Dict[str, Dict[str, Dict[str, object]]] = {}
    doctrine_id: Optional[str] = None
    rule_id: Optional[str] = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError("Indentation must be multiples of two spaces")

        if indent == 0:
            doctrine_id = stripped.rstrip(":")
            doctrines[doctrine_id] = {}
            rule_id = None
            continue

        if indent == 2:
            if doctrine_id is None:
                raise ValueError("Doctrine rule defined before doctrine header")
            rule_id = stripped.rstrip(":")
            doctrines[doctrine_id][rule_id] = {}
            continue

        if doctrine_id is None or rule_id is None:
            raise ValueError("Attribute defined without active doctrine and rule")

        if indent == 4 and stripped.startswith("conditions"):
            doctrines[doctrine_id][rule_id]["conditions"] = []
            continue

        if indent == 4:
            key, _, value = stripped.partition(":")
            doctrines[doctrine_id][rule_id][key.strip()] = _parse_scalar(value.strip())
            continue

        if indent == 6 and stripped.startswith("- "):
            conditions = doctrines[doctrine_id][rule_id].setdefault("conditions", [])
            if not isinstance(conditions, list):
                raise ValueError("Conditions block must be a list")
            conditions.append(stripped[2:].strip())
            continue

        raise ValueError(f"Unsupported doctrine line: {raw_line}")

    return doctrines


def _parse_scalar(value: str) -> object:
    """Convert scalar strings to floats where applicable."""

    if value == "":
        return ""
    try:
        return float(value)
    except ValueError:
        return value
