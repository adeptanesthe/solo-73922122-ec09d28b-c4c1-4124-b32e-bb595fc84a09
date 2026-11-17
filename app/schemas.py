"""请求/响应模型与输入校验。"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_TOTAL_TOKENS = 512
MAX_CHUNK_SIZE = 32
MIN_CHUNK_SIZE = 1
MAX_S = 1000.0
HEAD_DIM = 16


class VerifyRequest(BaseModel):
    """/verify 请求体。"""

    seed: int = Field(..., ge=0, description="随机种子（非负整数）")
    doc_lengths: list[int] = Field(
        ..., min_length=1, description="各文档长度（正整数），总长不超过 512"
    )
    chunk_size: int = Field(..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    s: float = Field(..., gt=0.0, le=MAX_S, description="注意力缩放系数")

    @field_validator("doc_lengths")
    @classmethod
    def _check_doc_lengths(cls, value: list[int]) -> list[int]:
        if any(not isinstance(x, int) or x <= 0 for x in value):
            raise ValueError("doc_lengths 必须为正整数列表")
        total = sum(value)
        if total == 0 or total > MAX_TOTAL_TOKENS:
            raise ValueError(
                f"文档总长度必须在 1..{MAX_TOTAL_TOKENS} 之间，当前为 {total}"
            )
        return value

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("s 必须为有限正数")
        return value


class ErrorMetrics(BaseModel):
    output_max_abs_err: float
    grad_q_max_abs_err: float
    grad_k_max_abs_err: float
    grad_v_max_abs_err: float


class VerifyResponse(BaseModel):
    passed: bool
    tolerance: float
    errors: ErrorMetrics
    scale_used: float = Field(..., description="实际进入 softmax 的系数 s/4")
    total_tokens: int
    num_docs: int
    chunk_size: int
    s: float
    seed: int
    finite_ok: bool = Field(..., description="所有输出与梯度是否均为有限值")
    failure_reasons: list[str]
    output: list[list[float]] | None = None
    grad_q: list[list[float]] | None = None
    grad_k: list[list[float]] | None = None
    grad_v: list[list[float]] | None = None


class ScenarioSummary(BaseModel):
    name: str
    description: str
    request: VerifyRequest
