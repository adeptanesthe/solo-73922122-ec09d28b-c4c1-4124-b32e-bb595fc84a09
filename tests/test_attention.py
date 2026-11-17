"""分块注意力核心功能测试：正确性、gradcheck、边界情形、内存与源码约束。"""

from __future__ import annotations

import inspect

import pytest
import torch

from app.attention import HEAD_DIM, ChunkedCausalAttention, chunked_causal_attention
from app.reference import dense_reference, dense_reference_grads

torch.manual_seed(0)


def _make_batch(doc_lengths: list[int], seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = sum(doc_lengths)
    q = torch.randn(n, HEAD_DIM, generator=g, dtype=torch.float64)
    k = torch.randn(n, HEAD_DIM, generator=g, dtype=torch.float64)
    v = torch.randn(n, HEAD_DIM, generator=g, dtype=torch.float64)
    go = torch.randn(n, HEAD_DIM, generator=g, dtype=torch.float64)
    doc_ids = torch.cat(
        [torch.full((length,), i, dtype=torch.int64) for i, length in enumerate(doc_lengths)]
    )
    return q, k, v, go, doc_ids


def _run_pair(doc_lengths, chunk_size, s, seed=0):
    q, k, v, go, doc_ids = _make_batch(doc_lengths, seed)
    scale = s / 4.0

    qc, kc, vc = (t.clone().requires_grad_(True) for t in (q, k, v))
    out_c = chunked_causal_attention(qc, kc, vc, doc_ids, scale, chunk_size)
    gq_c, gk_c, gv_c = torch.autograd.grad(out_c, (qc, kc, vc), grad_outputs=go)

    out_r, gq_r, gk_r, gv_r = dense_reference_grads(q, k, v, doc_ids, scale, go)
    return (out_c, gq_c, gk_c, gv_c), (out_r, gq_r, gk_r, gv_r)


@pytest.mark.parametrize(
    "doc_lengths,chunk_size,s",
    [
        ([8], 1, 1.0),
        ([8], 4, 1.0),
        ([8], 8, 1.0),
        ([8], 32, 1.0),  # 块长大于总长
        ([7, 11, 5], 8, 1.0),  # 文档跨块
        ([16], 4, 2.5),  # 存在整块屏蔽
        ([33, 31], 32, 1000.0),  # 大分数
        ([100, 100, 100, 100, 112], 7, 999.0),  # 综合压力（N=512）
        ([1, 1], 32, 0.5),  # 最小批次
        ([3, 1, 9, 2], 5, 37.7),
        ([512], 32, 500.0),  # 单文档上限
    ],
)
def test_forward_and_grad_match_dense(doc_lengths, chunk_size, s):
    chunked, ref = _run_pair(doc_lengths, chunk_size, s)
    names = ["output", "grad_q", "grad_k", "grad_v"]
    for name, a, b in zip(names, chunked, ref):
        assert torch.isfinite(a).all(), f"{name} 含非有限值"
        err = torch.max(torch.abs(a - b)).item()
        assert err <= 1e-8, f"{name} 误差 {err:.3e} 超过 1e-8"


def test_gradcheck_forward_backward_formulas():
    """用有限差分核对自定义算子的前向与一阶反传公式。"""
    q, k, v, _, doc_ids = _make_batch([5, 3], seed=11)
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda qq, kk, vv: ChunkedCausalAttention.apply(qq, kk, vv, doc_ids, 0.75, 4),
        (q, k, v),
        eps=1e-7,
        atol=1e-6,
        rtol=1e-4,
    )


def test_chunk_size_does_not_change_result():
    """同批数据在不同块长下结果必须一致（切块不改变归一化/梯度）。"""
    q, k, v, go, doc_ids = _make_batch([13, 17, 7], seed=5)
    scale = 3.0 / 4.0
    outs, grads = [], []
    for chunk in (1, 2, 3, 7, 16, 32):
        qc, kc, vc = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = chunked_causal_attention(qc, kc, vc, doc_ids, scale, chunk)
        gq, gk, gv = torch.autograd.grad(out, (qc, kc, vc), grad_outputs=go)
        outs.append(out.detach())
        grads.append((gq.detach(), gk.detach(), gv.detach()))
    for other in outs[1:]:
        assert torch.allclose(outs[0], other, atol=1e-12, rtol=0)
    for go_tuple in grads[1:]:
        for base, other in zip(grads[0], go_tuple):
            assert torch.allclose(base, other, atol=1e-12, rtol=0)


def test_determinism_same_seed_same_result():
    a, _ = _run_pair([7, 11, 5], 8, 1.0, seed=42)
    b, _ = _run_pair([7, 11, 5], 8, 1.0, seed=42)
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb)


def test_mask_semantics_doc_local_causal():
    """直接验证输出语义：只能读取同文档不晚于自身的 token。"""
    q, k, v, _, doc_ids = _make_batch([4, 3], seed=3)
    out = chunked_causal_attention(q, k, v, doc_ids, 1.0, 3)

    n = len(doc_ids)
    pos = torch.arange(n)
    valid = (doc_ids[:, None] == doc_ids[None, :]) & (pos[None, :] <= pos[:, None])
    scores = (q @ k.T) * 1.0
    scores = scores.masked_fill(~valid, float("-inf"))
    probs = torch.softmax(scores, dim=1)
    expected = probs @ v
    assert torch.allclose(out, expected, atol=1e-12, rtol=0)


def test_no_nan_large_score_and_full_mask_rows():
    # s=1000（scale=250）：分数极大，前后向均不得出现 NaN/Inf。
    out_c, gq, gk, gv = _run_pair([33, 31], 32, 1000.0)[0]
    for t in (out_c, gq, gk, gv):
        assert torch.isfinite(t).all()

    # 大分数下概率应高度集中：对照小 s，最大概率显著增大。
    q, k, v, _, doc_ids = _make_batch([33, 31], seed=123)
    n = len(doc_ids)
    pos = torch.arange(n)
    valid = (doc_ids[:, None] == doc_ids[None, :]) & (pos[None, :] <= pos[:, None])

    def max_prob(scale: float) -> torch.Tensor:
        sc = ((q @ k.T) * scale).masked_fill(~valid, float("-inf"))
        return torch.softmax(sc, dim=1).max(dim=1).values

    assert max_prob(1000.0 / 4.0).mean() > 0.99
    assert max_prob(1.0 / 4.0).mean() < max_prob(1000.0 / 4.0).mean()


def test_saved_tensors_have_linear_footprint():
    """跨前后向保存量审计：不得出现 N x N（或更大）的二次规模张量。"""
    q, k, v, _, doc_ids = _make_batch([60, 40], seed=9)  # N=100
    n = q.shape[0]
    saved: list[torch.Tensor] = []

    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (saved.append(t), t)[1], lambda t: t
    ):
        qc = q.clone().requires_grad_(True)
        kc = k.clone().requires_grad_(True)
        vc = v.clone().requires_grad_(True)
        out = chunked_causal_attention(qc, kc, vc, doc_ids, 0.5, 16)
        out.sum().backward()

    assert saved, "自定义算子未保存任何张量"
    cap = n * HEAD_DIM + n  # O(N*d + N)，d=16 为常数
    for t in saved:
        assert t.numel() <= cap, (
            f"保存张量 {tuple(t.shape)} 规模 {t.numel()} 超过线性上限 {cap}"
        )
        # 明确禁止任何 N x N 级别的二维矩阵
        assert not (len(t.shape) == 2 and t.shape[0] == n and t.shape[1] > HEAD_DIM)


def test_no_builtin_attention_operator_in_source():
    """源码审计：分块算子不得调用任何现成注意力算子。"""
    source = inspect.getsource(__import__("app.attention", fromlist=["_"]))
    forbidden = [
        "scaled_dot_product_attention",
        "MultiheadAttention",
        "nn.functional.attention",
        "flash_attn",
        "torch.softmax",  # 分块实现必须手写 online softmax
    ]
    for token in forbidden:
        assert token not in source, f"分块算子源码中出现禁用 API：{token}"


def test_dense_reference_independent_and_valid():
    q, k, v, _, doc_ids = _make_batch([6, 4], seed=2)
    out = dense_reference(q, k, v, doc_ids, 0.25)
    assert out.shape == (10, HEAD_DIM)
    assert torch.isfinite(out).all()
    # 与朴素逐行 softmax 计算一致
    n = len(doc_ids)
    pos = torch.arange(n)
    valid = (doc_ids[:, None] == doc_ids[None, :]) & (pos[None, :] <= pos[:, None])
    p = torch.softmax(((q @ k.T) * 0.25).masked_fill(~valid, float("-inf")), dim=1)
    assert torch.allclose(out, p @ v, atol=1e-12, rtol=0)
