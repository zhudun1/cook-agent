# app/prompts/registry.py
"""
Prompt 版本管理与灰度路由（Prompt Registry）

生产级 Prompt 治理：
- 同一 prompt 可注册多个版本（content + weight）
- 按用户稳定分流：同一 user_id 始终命中同一版本（可 A/B 实验、逐步灰度）
- 版本内容哈希校验，避免配置错误

接入点：AgentContextBuilder.build 解析 agent 的 system_prompt 时调用
`prompt_registry.get(agent_name, user_id, default=config.system_prompt)`。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptVersion:
    """一个 prompt 版本。"""

    name: str
    content: str
    version: str = "v1"
    weight: float = 1.0  # 灰度权重（占该 prompt 总权重的比例）
    description: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    content_hash: str = field(default="")

    def __post_init__(self):
        self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]


class PromptRegistry:
    """Prompt 版本注册表（按名称 + 灰度权重解析）。"""

    def __init__(self):
        self._prompts: Dict[str, List[PromptVersion]] = {}

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        content: str,
        version: str = "v1",
        weight: float = 1.0,
        description: str = "",
    ) -> PromptVersion:
        """
        注册一个 prompt 版本。

        Args:
            name: prompt 名称（通常为 agent 名）
            content: prompt 内容
            version: 版本号（同 name+version 重复注册会覆盖）
            weight: 灰度权重
            description: 版本说明

        Returns:
            注册的版本
        """
        pv = PromptVersion(
            name=name, content=content, version=version,
            weight=max(0.0, weight), description=description,
        )
        versions = self._prompts.setdefault(name, [])
        # 覆盖同版本
        for i, v in enumerate(versions):
            if v.version == version:
                versions[i] = pv
                return pv
        versions.append(pv)
        return pv

    def unregister(self, name: str, version: Optional[str] = None) -> bool:
        """注销 prompt（全部或指定版本）。"""
        if name not in self._prompts:
            return False
        if version is None:
            del self._prompts[name]
            return True
        versions = self._prompts[name]
        self._prompts[name] = [v for v in versions if v.version != version]
        return True

    # ------------------------------------------------------------------
    def get(
        self,
        name: str,
        user_id: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        解析 prompt（按用户稳定灰度分流）。

        Args:
            name: prompt 名称
            user_id: 用户 ID（稳定分流依据；None 时命中默认版本）
            default: 未注册时返回的默认内容

        Returns:
            命中版本的 prompt 内容
        """
        versions = self._prompts.get(name)
        if not versions:
            return default or ""

        # 单一版本直接返回
        if len(versions) == 1:
            return versions[0].content

        # 多版本：按权重区间 + user_id 哈希稳定分流
        total = sum(v.weight for v in versions)
        if total <= 0:
            return versions[0].content

        if user_id is None:
            # 无用户：取权重最大版本（默认）
            return max(versions, key=lambda v: v.weight).content

        # user_id -> [0, 1) 稳定哈希
        h = int(hashlib.md5(user_id.encode("utf-8")).hexdigest()[:8], 16)
        point = (h % 10000) / 10000.0

        acc = 0.0
        for v in versions:
            acc += v.weight / total
            if point < acc:
                return v.content
        return versions[-1].content

    def list_versions(self, name: str) -> List[dict]:
        """列出某 prompt 的全部版本。"""
        versions = self._prompts.get(name, [])
        return [
            {
                "name": v.name,
                "version": v.version,
                "weight": v.weight,
                "description": v.description,
                "content_hash": v.content_hash,
                "created_at": v.created_at,
            }
            for v in versions
        ]

    def clear(self) -> None:
        self._prompts.clear()


# 全局单例
prompt_registry = PromptRegistry()
