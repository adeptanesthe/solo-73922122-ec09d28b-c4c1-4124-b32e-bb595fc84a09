"""数据生成与分块/稠密实现的核对逻辑。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention import HEAD_DIM, chunked_causal_attention
from .reference import dense_reference_grads
from .schemas import VerifyRequest, VerifyResponse

TOLERANCE = 1e-8


@dataclass
class GeneratedData:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    doc_ids: torch.Tensor
    grad_upstream: torch.Tensor


def make_doc_ids(doc_lengths: list[int]) -> torch.Tensor:
    """把文档长度列表展开为每个 token 的文档编号。"""
    ids: list[int] = []
    for doc_id, length in enumerate(doc_lengths):
        ids.extend([doc_id] * length)
    return torch.tensor(ids, dtype=torch.int64)


def generate_inputs(req: VerifyRequest) -> GeneratedData:
    """按种子确定性生成 float64 的 Q/K/V 与上游梯度（16 维）。"""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(req.seed))
    n = sum(req.doc_lengths)

    def randn() -> torch.Tensor:
        return torch.randn(n, HEAD_DIM, generator=g, dtype=torch.float64)

    q, k, v = randn(), randn(), randn()
    grad_upstream = randn()
    return GeneratedData(
        q=q, k=k, v=v, doc_ids=make_doc_ids(req.doc_lengths), grad_upstream=grad_upstream
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def run_verification(req: VerifyRequest, include_tensors: bool = True) -> VerifyResponse:
    """生成本批次数据，分别走分块算子与稠密参考，比较前向与梯度。"""
    data = generate_inputs(req)
    # 题目规定 softmax(s QK^T / 4) V。
    scale = float(req.s) / 4.0

    # --- 分块自定义算子（前向 + 反向） ---
    q_c = data.q.detach().clone().requires_grad_(True)
    k_c = data.k.detach().clone().requires_grad_(True)
    v_c = data.v.detach().clone().requires_grad_(True)
    out_c = chunked_causal_attention(
        q_c, k_c, v_c, data.doc_ids, scale, req.chunk_size
    )
    gq_c, gk_c, gv_c = torch.autograd.grad(
        out_c, (q_c, k_c, v_c), grad_outputs=data.grad_upstream
    )

    # --- 独立稠密参考（前向 + 反向） ---
    out_r, gq_r, gk_r, gv_r = dense_reference_grads(
        data.q, data.k, data.v, data.doc_ids, scale, data.grad_upstream
    )

    errors = {
        "output_max_abs_err": _max_abs(out_c, out_r),
        "grad_q_max_abs_err": _max_abs(gq_c, gq_r),
        "grad_k_max_abs_err": _max_abs(gk_c, gk_r),
        "grad_v_max_abs_err": _max_abs(gv_c, gv_r),
    }
    finite_tensors = [out_c, gq_c, gk_c, gv_c]
    finite_ok = all(torch.isfinite(t).all().item() for t in finite_tensors)

    failure_reasons: list[str] = []
    for name, value in errors.items():
        if value > TOLERANCE:
            failure_reasons.append(f"{name}={value:.3e} 超过容差 {TOLERANCE:.0e}")
        if value != value:  # NaN
            failure_reasons.append(f"{name} 为 NaN")
    if not finite_ok:
        failure_reasons.append("分块实现的输出或梯度中存在 NaN/Inf")

    passed = not failure_reasons

    def to_list(t: torch.Tensor) -> list[list[float]]:
        return t.detach().tolist()

    return VerifyResponse(
        passed=passed,
        tolerance=TOLERANCE,
        errors=errors,
        scale_used=scale,
        total_tokens=sum(req.doc_lengths),
        num_docs=len(req.doc_lengths),
        chunk_size=req.chunk_size,
        s=req.s,
        seed=req.seed,
        finite_ok=finite_ok,
        failure_reasons=failure_reasons,
        output=to_list(out_c) if include_tensors else None,
        grad_q=to_list(gq_c) if include_tensors else None,
        grad_k=to_list(gk_c) if include_tensors else None,
        grad_v=to_list(gv_c) if include_tensors else None,
    )
