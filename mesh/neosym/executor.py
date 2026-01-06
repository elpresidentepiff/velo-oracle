"""Runtime executor for compiled doctrine graphs."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

from .compiler import DoctrineGraph, DoctrineRule


@dataclass(slots=True)
class RuleEvaluation:
    """Evaluation result for a single doctrine rule."""

    rule: DoctrineRule
    satisfied: bool
    contribution: float


@dataclass(slots=True)
class DoctrineEvaluation:
    """Aggregated evaluation for a doctrine."""

    doctrine_id: str
    score: float
    rules: Sequence[RuleEvaluation]


class DoctrineExecutor:
    """Evaluates doctrine graphs against feature dictionaries."""

    def __init__(self, graphs: Mapping[str, DoctrineGraph]) -> None:
        self._graphs = graphs

    def evaluate(
        self,
        features: Mapping[str, float],
        extra_context: Optional[Mapping[str, float]] = None,
    ) -> Mapping[str, DoctrineEvaluation]:
        """Evaluate all doctrine graphs and return per-doctrine scores."""

        context: MutableMapping[str, float] = dict(features)
        if extra_context:
            context.update(extra_context)

        results: Dict[str, DoctrineEvaluation] = {}
        for doctrine_id, graph in self._graphs.items():
            rule_evaluations: Sequence[RuleEvaluation] = tuple(
                self._evaluate_rule(rule, context) for rule in graph.rules
            )
            score = max((evaluation.contribution for evaluation in rule_evaluations), default=0.0)
            results[doctrine_id] = DoctrineEvaluation(
                doctrine_id=doctrine_id,
                score=score,
                rules=rule_evaluations,
            )
        return results

    def _evaluate_rule(self, rule: DoctrineRule, context: Mapping[str, float]) -> RuleEvaluation:
        satisfied = all(_safe_eval(condition.expression, context) for condition in rule.conditions)
        contribution = rule.score if satisfied else 0.0
        return RuleEvaluation(rule=rule, satisfied=satisfied, contribution=contribution)


def _safe_eval(expression: str, context: Mapping[str, float]) -> bool:
    """Evaluate a boolean expression in a controlled environment."""

    node = ast.parse(expression, mode="eval")
    return bool(_eval_node(node.body, context))


def _eval_node(node: ast.AST, context: Mapping[str, float]) -> float:
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(value, context) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(bool(value) for value in values)
        if isinstance(node.op, ast.Or):
            return any(bool(value) for value in values)
        raise ValueError(f"Unsupported boolean operator: {node.op}")

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context)
        result = True
        for operator, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, context)
            if isinstance(operator, ast.Gt):
                result = result and left > right
            elif isinstance(operator, ast.GtE):
                result = result and left >= right
            elif isinstance(operator, ast.Lt):
                result = result and left < right
            elif isinstance(operator, ast.LtE):
                result = result and left <= right
            elif isinstance(operator, ast.Eq):
                result = result and left == right
            elif isinstance(operator, ast.NotEq):
                result = result and left != right
            else:
                raise ValueError(f"Unsupported comparison operator: {operator}")
            left = right
        return result

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ValueError(f"Unsupported binary operator: {node.op}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, context)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError(f"Unsupported unary operator: {node.op}")

    if isinstance(node, ast.Name):
        return float(context.get(node.id, 0.0))

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float, bool)):
            return float(value)
        raise ValueError(f"Unsupported constant type: {type(value)!r}")

    raise ValueError(f"Unsupported expression node: {node!r}")
