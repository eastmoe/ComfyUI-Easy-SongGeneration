"""
Small OmegaConf compatibility subset for inference configs.

This project only needs static config loading, attribute-style access, nested
containers, and ``OmegaConf.to_container/create``.  Keeping that subset local
removes the runtime dependency on omegaconf/antlr while preserving the existing
call sites.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from collections.abc import Iterable, MutableMapping
from pathlib import Path
from typing import Any, Callable


_MISSING = object()


def _wrap(value: Any) -> Any:
    if isinstance(value, DictConfig) or isinstance(value, ListConfig):
        return value
    if isinstance(value, dict):
        return DictConfig(value)
    if isinstance(value, list):
        return ListConfig(value)
    return value


def _unwrap(value: Any, *, resolve: bool = False, root: Any = None) -> Any:
    if isinstance(value, DictConfig):
        return {k: _unwrap(v, resolve=resolve, root=root or value) for k, v in value.items()}
    if isinstance(value, ListConfig):
        return [_unwrap(v, resolve=resolve, root=root) for v in value]
    if resolve and isinstance(value, str):
        return OmegaConf._resolve_string(value, root)
    return value


class DictConfig(MutableMapping):
    def __init__(self, value: dict[str, Any] | None = None):
        object.__setattr__(self, "_data", {})
        for key, item in (value or {}).items():
            self._data[key] = _wrap(item)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = _wrap(value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._data[key] = _wrap(value)

    def __repr__(self) -> str:
        return f"DictConfig({self._data!r})"


class ListConfig(list):
    def __init__(self, values: Iterable[Any] = ()):
        super().__init__(_wrap(v) for v in values)

    def append(self, value: Any) -> None:
        super().append(_wrap(value))

    def extend(self, values: Iterable[Any]) -> None:
        super().extend(_wrap(v) for v in values)

    def insert(self, index: int, value: Any) -> None:
        super().insert(index, _wrap(value))

    def __setitem__(self, index, value: Any) -> None:
        if isinstance(index, slice):
            super().__setitem__(index, [_wrap(v) for v in value])
        else:
            super().__setitem__(index, _wrap(value))


class OmegaConf:
    _resolvers: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register_new_resolver(cls, name: str, resolver: Callable[..., Any]) -> None:
        cls._resolvers[name] = resolver

    @staticmethod
    def create(value: Any) -> Any:
        return _wrap(value)

    @staticmethod
    def to_container(value: Any, resolve: bool = False) -> Any:
        return _unwrap(value, resolve=resolve, root=value)

    @classmethod
    def load(cls, path: str | Path) -> Any:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(text)
        elif suffix == ".toml":
            data = tomllib.loads(text)
        elif suffix in {".yaml", ".yml"}:
            data = _load_yaml(text, path)
        else:
            raise ValueError(f"Unsupported config format: {path}")
        return _wrap(data)

    @classmethod
    def _resolve_string(cls, value: str, root: Any) -> Any:
        pattern = re.compile(r"\$\{([^{}]+)\}")
        while True:
            match = re.fullmatch(pattern, value)
            if match:
                return cls._resolve_expression(match.group(1), root)

            if not pattern.search(value):
                return value

            def replace(match_obj: re.Match[str]) -> str:
                return str(cls._resolve_expression(match_obj.group(1), root))

            value = pattern.sub(replace, value)

    @classmethod
    def _resolve_expression(cls, expr: str, root: Any) -> Any:
        name, sep, arg_text = expr.partition(":")
        if sep and name in cls._resolvers:
            args = _parse_resolver_args(arg_text)
            resolved_args = [cls._resolve_string(arg, root) if isinstance(arg, str) else arg for arg in args]
            return cls._resolvers[name](*resolved_args)
        return _select(root, expr)


def _select(root: Any, dotted_path: str) -> Any:
    cur = root
    for part in dotted_path.split("."):
        if isinstance(cur, DictConfig):
            cur = cur[part]
        elif isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, (ListConfig, list)):
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _parse_resolver_args(text: str) -> list[Any]:
    if not text:
        return []
    try:
        value = ast.literal_eval(f"({text},)")
        return list(value)
    except (SyntaxError, ValueError):
        return [part.strip() for part in text.split(",") if part.strip()]


def _load_yaml(text: str, path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_simple_yaml(text)
    return yaml.safe_load(text)  # type: ignore[no-any-return]


def _strip_comment(line: str) -> str:
    quote: str | None = None
    for idx, char in enumerate(line):
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        elif char == "#" and quote is None:
            return line[:idx].rstrip()
    return line.rstrip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value[0], value[-1]) in {("'", "'"), ('"', '"')}:
        return ast.literal_eval(value)
    if value[0] in "[{" and value[-1] in "]}":
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            if value[0] == "[":
                return [_parse_scalar(item) for item in _split_flow_items(value[1:-1])]
            if value[0] == "{":
                return _parse_flow_mapping(value[1:-1])
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_simple_yaml(text: str) -> Any:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        lines.append((len(line) - len(line.lstrip(" ")), line.strip()))
    if not lines:
        return None
    value, index = _parse_yaml_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError("Unsupported YAML structure in static config")
    return value


def _split_flow_items(text: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, char in enumerate(text):
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        elif quote is None:
            if char in "[{(":
                depth += 1
            elif char in "]})":
                depth -= 1
            elif char == "," and depth == 0:
                item = text[start:index].strip()
                if item:
                    items.append(item)
                start = index + 1
    item = text[start:].strip()
    if item:
        items.append(item)
    return items


def _parse_flow_mapping(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in _split_flow_items(text):
        if ":" not in item:
            raise ValueError(f"Unsupported YAML mapping item: {item}")
        key, value = item.split(":", 1)
        parsed[key.strip().strip("'\"")] = _parse_scalar(value)
    return parsed


def _parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    is_list = lines[index][1].startswith("- ")
    result: Any = [] if is_list else {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError("Unsupported YAML indentation")

        if is_list:
            if not content.startswith("- "):
                break
            item_text = content[2:].strip()
            if item_text:
                result.append(_parse_scalar(item_text))
                index += 1
            else:
                item, index = _parse_yaml_block(lines, index + 1, _next_indent(lines, index))
                result.append(item)
        else:
            if ":" not in content:
                raise ValueError(f"Unsupported YAML line: {content}")
            key, value_text = content.split(":", 1)
            key = key.strip()
            value_text = value_text.strip()
            if value_text:
                result[key] = _parse_scalar(value_text)
                index += 1
            else:
                child_indent = _next_indent(lines, index)
                if child_indent <= indent:
                    result[key] = None
                    index += 1
                else:
                    result[key], index = _parse_yaml_block(lines, index + 1, child_indent)
    return result, index


def _next_indent(lines: list[tuple[int, str]], index: int) -> int:
    if index + 1 >= len(lines):
        return lines[index][0]
    return lines[index + 1][0]
