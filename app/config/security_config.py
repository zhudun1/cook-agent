# app/config/security_config.py
"""
P0 安全配置：成本熔断、人工介入审批、工具权限矩阵、注入纵深防御。

YAML 配置段: security
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CostGuardConfig(BaseModel):
    """成本熔断配置。"""

    enabled: bool = True
    # 单会话累计 token 预算（输入+输出，超过触发动作）
    session_token_budget: int = Field(default=50000, ge=0)
    # 单轮（一次 LLM 调用）token 预算
    per_turn_token_budget: int = Field(default=12000, ge=0)
    # 超过预算的动作: degrade（降级到低成本层级）/ refuse（拒绝调用）/ warn（仅告警）
    action: str = Field(default="degrade", pattern="^(degrade|refuse|warn)$")
    # 达到预算 warn_threshold_ratio 比例时先告警
    warn_threshold_ratio: float = Field(default=0.8, ge=0.0, le=1.0)


class ApprovalConfig(BaseModel):
    """人工介入（HITL）审批配置。"""

    enabled: bool = True
    # 需要审批的工具名（支持 glob，如 "image_generator", "delete_*"）
    required_tool_patterns: List[str] = Field(
        default_factory=lambda: ["image_generator"]
    )
    # 审批等待超时（秒），超时按拒绝处理
    timeout_seconds: float = Field(default=120.0, gt=0)
    # admin 用户自动通过审批
    auto_approve_admin: bool = True
    # admin 用户列表（与 permissions.admin_users 保持一致）
    admin_users: List[str] = Field(default_factory=list)


class PermissionConfig(BaseModel):
    """工具权限矩阵配置。"""

    enabled: bool = True
    # admin 用户（user_id），拥有全部工具权限
    admin_users: List[str] = Field(default_factory=list)
    # 敏感工具：默认拒绝（除非显式 allow）
    default_denied_tools: List[str] = Field(default_factory=list)
    # 显式放行：user_id -> [tool 名或 glob]
    allow: dict = Field(default_factory=dict)
    # 显式拒绝：user_id -> [tool 名或 glob]
    deny: dict = Field(default_factory=dict)


class InjectionGuardConfig(BaseModel):
    """工具返回内容注入检测（纵深防御第二层）。"""

    enabled: bool = True
    # 是否检查工具返回内容（第一层是用户输入检查）
    check_tool_results: bool = True
    # 命中时动作: block（截断风险内容）/ warn（仅告警）
    action: str = Field(default="block", pattern="^(block|warn)$")
    # 追加的注入模式（正则），与内置模式合并
    extra_patterns: List[str] = Field(default_factory=list)


class SecurityConfig(BaseModel):
    """P0 安全总配置。"""

    cost_guard: CostGuardConfig = Field(default_factory=CostGuardConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    injection_guard: InjectionGuardConfig = Field(default_factory=InjectionGuardConfig)
