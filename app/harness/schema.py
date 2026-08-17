# app/harness/schema.py
"""
结构化结果校验（JSON Schema 子集）

借鉴 DeepSeek Harness 的 schema 化子代理结果：
工具/子代理返回的对象先经过 schema 校验，失败即失败，
不让脏数据流入后续阶段（fail fast）。

支持的 Schema 关键字（子集）:
- type: object | string | number | integer | boolean | array | null
- properties / required / additionalProperties（object）
- items（array 元素 schema）
- enum / const / oneOf

校验结果统一为 `ValidationResult(valid, errors)`，
errors 为人类可读的错误路径列表（如 "properties.name: 期望 string, 实际 integer"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# Schema 关键字白名单（防止 schema 本身被滥用为代码执行载体）
ALLOWED_KEYWORDS = {
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "const", "oneOf",
}


class SchemaError(ValueError):
    """Schema 本身非法。"""


@dataclass
class ValidationResult:
    """校验结果。"""

    valid: bool
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"valid": self.valid, "errors": self.errors}


def validate_schema(schema: Dict[str, Any]) -> None:
    """校验 schema 结构本身是否合法（仅允许白名单关键字）。"""
    if not isinstance(schema, dict):
        raise SchemaError("schema must be an object")
    if "type" in schema and schema["type"] not in (
        "object", "string", "number", "integer", "boolean", "array", "null",
    ):
        raise SchemaError(f"unsupported type: {schema['type']}")
    for key in schema:
        if key not in ALLOWED_KEYWORDS:
            raise SchemaError(f"unsupported keyword: {key}")
    if schema.get("type") == "object":
        props = schema.get("properties")
        if props is not None and not isinstance(props, dict):
            raise SchemaError("properties must be an object")
    if schema.get("type") == "array" and "items" in schema:
        validate_schema(schema["items"])


def validate(obj: Any, schema: Dict[str, Any], _path: str = "$") -> List[str]:
    """
    校验对象是否符合 schema。

    Args:
        obj: 待校验对象
        schema: JSON Schema 子集
        _path: 当前校验路径（内部递归用）

    Returns:
        错误列表（空表示通过）
    """
    validate_schema(schema)
    return _validate(obj, schema, _path)


def _validate(obj: Any, schema: Dict[str, Any], path: str) -> List[str]:
    errors: List[str] = []

    # const / enum
    if "const" in schema:
        if obj != schema["const"]:
            errors.append(f"{path}: 期望常量 {schema['const']!r}, 实际 {obj!r}")
        return errors
    if "enum" in schema:
        if obj not in schema["enum"]:
            errors.append(f"{path}: 值 {obj!r} 不在枚举 {schema['enum']} 中")
        return errors

    # oneOf
    if "oneOf" in schema:
        matched = 0
        for sub in schema["oneOf"]:
            if not _validate(obj, sub, path):
                matched += 1
        if matched != 1:
            errors.append(f"{path}: oneOf 匹配 {matched} 个分支（期望恰好 1 个）")
        return errors

    # type
    expected = schema.get("type")
    if expected is not None:
        type_ok = _type_matches(obj, expected)
        if not type_ok:
            errors.append(f"{path}: 期望类型 {expected}, 实际 {type(obj).__name__}")
            return errors

    # 结构约束
    if expected == "object" or isinstance(obj, dict):
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        additional = schema.get("additionalProperties", True)

        for name in required:
            if name not in obj:
                errors.append(f"{path}.{name}: 缺少必填字段")

        for key, value in (obj.items() if isinstance(obj, dict) else []):
            if key in props:
                errors.extend(_validate(value, props[key], f"{path}.{key}"))
            elif not additional:
                errors.append(f"{path}.{key}: 不允许的额外字段")

    if expected == "array" and isinstance(obj, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(obj):
                errors.extend(_validate(item, items_schema, f"{path}[{i}]"))

    return errors


def _type_matches(obj: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(obj, dict)
    if expected == "string":
        return isinstance(obj, str)
    if expected == "number":
        return isinstance(obj, (int, float)) and not isinstance(obj, bool)
    if expected == "integer":
        return isinstance(obj, int) and not isinstance(obj, bool)
    if expected == "boolean":
        return isinstance(obj, bool)
    if expected == "array":
        return isinstance(obj, list)
    if expected == "null":
        return obj is None
    return True


def validate_result(obj: Any, schema: Dict[str, Any]) -> ValidationResult:
    """
    便捷入口：校验并返回 ValidationResult。

    用法::

        result = validate_result({"name": "红烧肉"}, {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        })
        if not result.valid:
            raise ValueError("; ".join(result.errors))
    """
    try:
        errors = validate(obj, schema)
        return ValidationResult(valid=not errors, errors=errors)
    except SchemaError as e:
        return ValidationResult(valid=False, errors=[str(e)])
