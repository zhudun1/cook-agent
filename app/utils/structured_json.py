import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# JSON extraction regex
JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r"\{[\s\S]*?\}", re.MULTILINE)


# ---------------------------------------------------------------------------
# JSON 自动修复（兜底）
# ---------------------------------------------------------------------------

def repair_json_text(text: str) -> str:
    """
    修复 LLM 输出中常见的 JSON 格式问题：

    1. 移除 markdown 代码围栏
    2. 去掉前导/尾随非 JSON 噪音（保留最外层 {...} 或 [...]）
    3. 修复尾随逗号
    4. 修复单引号字符串 / 未加引号的键
    5. 修复被截断的 JSON（补全引号与括号）
    6. 修复 JS 风格的裸值（undefined/NaN/Infinity -> null）

    Args:
        text: 可能包含 JSON 的原始文本

    Returns:
        修复后的 JSON 字符串（尽力而为）
    """
    if not text:
        return text

    s = text.strip()

    # 1. 提取代码块内容
    block = JSON_BLOCK_RE.search(s)
    if block:
        s = block.group(1).strip()

    # 2. 提取最外层 JSON 结构（若周围有噪音文本）
    s = _extract_outer_json(s)

    if not s:
        return text

    # 3. 修复尾随逗号
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 4. 修复单引号字符串（仅当字符串内无双引号冲突时）
    s = _fix_single_quotes(s)

    # 5. 修复未加引号的键（{key: value} -> {"key": value}）
    s = _fix_unquoted_keys(s)

    # 6. 修复 JS 裸值
    s = re.sub(r":\s*undefined\b", ": null", s)
    s = re.sub(r":\s*NaN\b", ": null", s)
    s = re.sub(r":\s*Infinity\b", ": null", s)

    # 7. 修复截断：补齐未闭合的引号/括号
    s = _close_truncated(s)

    return s


def _extract_outer_json(s: str) -> str:
    """提取最外层 { ... } 或 [ ... ] 结构（含括号平衡）。"""
    start = -1
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start == -1:
        return ""
    opener = s[start]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_str = False
    escape = False
    end = len(s)
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return s[start:end]


def _fix_single_quotes(s: str) -> str:
    """把单引号字符串转为双引号（安全启发式）。"""
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "'":
            # 判断是否为字符串边界：前一个字符是结构符或行首，后一个不是引号
            prev = s[i - 1] if i > 0 else ""
            nxt = s[i + 1] if i + 1 < n else ""
            if prev in ":{[, " or prev == "":
                # 开引号：查找配对闭引号
                j = i + 1
                while j < n:
                    if s[j] == "\\":
                        j += 2
                        continue
                    if s[j] == "'":
                        break
                    if s[j] in "\"\n":
                        break
                    j += 1
                if j < n and s[j] == "'":
                    out.append('"')
                    out.append(s[i + 1 : j].replace('"', '\\"'))
                    out.append('"')
                    i = j + 1
                    continue
            elif nxt in ",}]: " or nxt == "":
                # 闭引号：前一字符属于字符串内容
                out.append('"')
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")


def _fix_unquoted_keys(s: str) -> str:
    """把未加引号的键（JSON5/JS 风格）转为双引号键。"""
    return _KEY_RE.sub(lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', s)


def _close_truncated(s: str) -> str:
    """修复截断：补齐未闭合字符串与括号。"""
    # 补全未闭合字符串
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
    if in_str:
        s += '"'

    # 补齐未闭合括号（按栈补全）
    stack = []
    pairs = {"{": "}", "[": "]"}
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in pairs:
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    for opener in reversed(stack):
        s += pairs[opener]
    return s


# ---------------------------------------------------------------------------
# 解析入口
# ---------------------------------------------------------------------------

def parse_json_auto(content: str) -> Dict[str, Any]:
    """
    解析 LLM 输出中的 JSON，带自动修复兜底。

    修复链路：直接解析 -> 代码块提取 -> JSON 对象模式 -> 修复后重试。

    Args:
        content: LLM 原始输出

    Returns:
        解析出的 JSON 对象

    Raises:
        ValueError: 所有策略均失败
    """
    attempts = [
        content,
        _first_block(content),
        _first_object_pattern(content),
        repair_json_text(content),
        repair_json_text(_first_block(content) or ""),
        repair_json_text(_first_object_pattern(content) or ""),
    ]

    seen = set()
    for candidate in attempts:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
            if isinstance(result, list) and result and isinstance(result[0], dict):
                return result[0]
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON found in response")


def _first_block(content: str) -> Optional[str]:
    """提取第一个 json 代码块内容。"""
    for match in JSON_BLOCK_RE.findall(content):
        return match
    return None


def _first_object_pattern(content: str) -> Optional[str]:
    """提取第一个 { ... } 片段。"""
    for match in JSON_OBJECT_RE.findall(content):
        return match
    return None


def extract_first_valid_json(content: str) -> Dict[str, Any]:
    """兼容旧接口：提取 LLM 输出中的第一个有效 JSON 对象。"""
    # Try to extract from code block first
    for match in JSON_BLOCK_RE.findall(content):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try to extract direct JSON object
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object pattern
    for match in JSON_OBJECT_RE.findall(content):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON found in response")
