"""进程内增量注意力会话 API 的请求/响应模型。"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schemas import HEAD_DIM, MAX_S

# 每个分支最多缓存的 token 数。
MAX_BRANCH_TOKENS = 512
# 单个 Q/K/V 分量的允许范围。
COMPONENT_ABS_LIMIT = 8.0
# 增量输出与全量重算的允许误差。
INCREMENTAL_TOLERANCE = 1e-8


def check_vector(value: list[float], field_name: str) -> list[float]:
    """校验 16 维向量：分量有限且落在 [-8, 8]。"""
    if len(value) != HEAD_DIM:
        raise ValueError(f"{field_name} 必须为 {HEAD_DIM} 维，当前为 {len(value)} 维")
    result: list[float] = []
    for index, component in enumerate(value):
        number = float(component)
        if not math.isfinite(number):
            raise ValueError(f"{field_name}[{index}] 必须为有限值")
        if number < -COMPONENT_ABS_LIMIT or number > COMPONENT_ABS_LIMIT:
            raise ValueError(
                f"{field_name}[{index}]={number} 超出 "
                f"[-{COMPONENT_ABS_LIMIT:g}, {COMPONENT_ABS_LIMIT:g}]"
            )
        result.append(number)
    return result


class _StrictRequest(BaseModel):
    """请求体拒绝未声明字段，避免客户端误传 s 等会话级参数。"""

    model_config = ConfigDict(extra="forbid")


class SessionCreateRequest(_StrictRequest):
    """创建会话：s 在此固定，后续所有分支共用。"""

    s: float = Field(..., gt=0.0, le=MAX_S, description="固定的注意力缩放系数 s")

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("s 必须为有限正数")
        return value


class AppendTokenRequest(_StrictRequest):
    """向分支末尾追加一个 token。"""

    expected_revision: int = Field(
        ..., ge=0, strict=True,
        description="调用方看到的该分支修订号；不匹配则拒绝",
    )
    q: list[float] = Field(..., description="16 维 query，分量在 [-8, 8]")
    k: list[float] = Field(..., description="16 维 key，分量在 [-8, 8]")
    v: list[float] = Field(..., description="16 维 value，分量在 [-8, 8]")
    new_document: bool = Field(
        ..., strict=True,
        description="是否以此 token 开启新文档；分支首个 token 必须为 true",
    )

    @field_validator("q")
    @classmethod
    def _check_q(cls, value: list[float]) -> list[float]:
        return check_vector(value, "q")

    @field_validator("k")
    @classmethod
    def _check_k(cls, value: list[float]) -> list[float]:
        return check_vector(value, "k")

    @field_validator("v")
    @classmethod
    def _check_v(cls, value: list[float]) -> list[float]:
        return check_vector(value, "v")


class ForkRequest(_StrictRequest):
    """从父支当前修订号分叉。"""

    expected_revision: int = Field(
        ..., ge=0, strict=True,
        description="父支当前修订号；必须与父支实际修订号一致",
    )


class BranchSummary(BaseModel):
    branch_id: str
    parent_id: str | None
    fork_revision: int | None = Field(
        None, description="分叉时父支的修订号；根支为 null"
    )
    revision: int = Field(
        ...,
        description=(
            "分支修订号，恒等于该支已提交 token 数；新根支为 0，"
            "分叉出的子支从父支分叉修订号起步，每次成功追加 +1"
        ),
    )
    token_count: int
    num_documents: int
    document_lengths: list[int]
    max_tokens: int
    remaining_tokens: int
    created_at: float


class SessionSummary(BaseModel):
    session_id: str
    s: float
    scale_used: float = Field(..., description="实际进入 softmax 的系数 s/4")
    max_tokens_per_branch: int
    branches: list[BranchSummary]
    created_at: float


class AppendTokenResponse(BaseModel):
    session_id: str
    branch_id: str
    previous_revision: int
    revision: int
    token_index: int = Field(..., description="新 token 在该分支中的全局下标")
    document_index: int
    document_length: int = Field(..., description="新 token 所在文档的当前长度")
    num_documents: int
    total_tokens: int
    s: float
    scale_used: float
    output: list[float] = Field(..., description="该新 token 的 16 维 float64 输出")


class ForkResponse(BaseModel):
    parent_branch_id: str
    parent_revision: int
    child: BranchSummary


class BranchVerifyResponse(BaseModel):
    branch_id: str
    revision: int
    passed: bool
    tolerance: float
    max_abs_err: float
    checked_tokens: int
    finite_ok: bool
    failure_reasons: list[str]
