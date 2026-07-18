from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class CourierRule(BaseModel):
    name: str
    pattern: str
    prefixes: list[str] = Field(default_factory=list)
    context_terms: list[str] = Field(default_factory=list)
    check_digit: str | None = None


class CourierRuleSet(BaseModel):
    version: str = "1"
    rules: list[CourierRule] = Field(default_factory=list)


@lru_cache(maxsize=4)
def load_courier_rules(path: str) -> CourierRuleSet:
    rule_path = Path(path)
    if not rule_path.is_file():
        return CourierRuleSet()
    return CourierRuleSet.model_validate_json(rule_path.read_text(encoding="utf-8"))


def matching_rules(value: str, path: str, context: str = "") -> list[str]:
    matched: list[str] = []
    upper_value = value.upper()
    lower_context = context.lower()
    for rule in load_courier_rules(path).rules:
        if rule.prefixes and not any(upper_value.startswith(prefix.upper()) for prefix in rule.prefixes):
            continue
        if not re.fullmatch(rule.pattern, upper_value):
            continue
        if rule.context_terms and context and not any(term.lower() in lower_context for term in rule.context_terms):
            continue
        matched.append(rule.name)
    return matched


def validate_rule_file(path: str) -> list[str]:
    errors: list[str] = []
    try:
        rules = load_courier_rules(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"Invalid courier rule file: {type(exc).__name__}"]
    names: set[str] = set()
    for rule in rules.rules:
        if rule.name in names:
            errors.append(f"Duplicate rule name: {rule.name}")
        names.add(rule.name)
        try:
            re.compile(rule.pattern)
        except re.error as exc:
            errors.append(f"Invalid regex for {rule.name}: {exc}")
    return errors
