"""分块（文档内因果）注意力：自定义 autograd 算子。

约定
----
* 所有计算均在 CPU、float64 下完成。
* token i 只能读取同一篇文档中位置不晚于自己的 token：
  ``valid(i, j) = (doc_id[i] == doc_id[j]) and (pos[j] <= pos[i])``。
* 注意力输出 ``O = softmax(s * Q K^T / 4) V``，s 为调用方传入的缩放系数。
* 前向、后向均按长度 C 的块逐块处理；单块工作张量为 O(C^2 + N) 量级，
  跨前后向保存的张量为 O(N) 量级（均不含与 N 无关的常数维度 d=16）。

实现只使用基本张量算子（逐点运算、矩阵乘、求和、比较等），
不调用任何框架内置的注意力算子，也不依赖框架自带的 softmax 原语。
"""

from __future__ import annotations

import torch

HEAD_DIM = 16
NEG_INF = torch.tensor(float("-inf"), dtype=torch.float64)


def _split(num: int, chunk: int) -> list[tuple[int, int]]:
    """返回左闭右开的分块区间列表。"""
    return [(b, min(b + chunk, num)) for b in range(0, num, chunk)]


def _block_valid(
    doc_ids: torch.Tensor, q0: int, q1: int, k0: int, k1: int
) -> torch.Tensor:
    """构造块 (q0:q1) x (k0:k1) 的文档内因果掩码。

    返回 bool 矩阵，True 表示该 (query, key) 对允许被读取。
    """
    same_doc = doc_ids[q0:q1, None] == doc_ids[None, k0:k1]
    k_pos = torch.arange(k0, k1, dtype=torch.int64)
    q_pos = torch.arange(q0, q1, dtype=torch.int64)
    causal = k_pos[None, :] <= q_pos[:, None]
    return same_doc & causal


class ChunkedCausalAttention(torch.autograd.Function):
    """分块文档内因果注意力的自定义 autograd 算子。

    前向使用 online softmax（按块维护行最大值与归一化分母），
    后向按同样的分块方式重算块内概率，避免保存 N x N 注意力矩阵。
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        doc_ids: torch.Tensor,
        scale: float,
        chunk_size: int,
    ) -> torch.Tensor:
        n, d = q.shape
        blocks = _split(n, chunk_size)
        device = q.device

        # 行级 running 统计量（online softmax），全程 O(N)。
        row_max = torch.full((n,), float("-inf"), dtype=torch.float64, device=device)
        row_denom = torch.zeros((n,), dtype=torch.float64, device=device)
        out = torch.zeros((n, d), dtype=torch.float64, device=device)

        for k0, k1 in blocks:
            # key 块起点必须早于 query 块终点（k0 < q1），否则块内全是未来
            # token，掩码将整行屏蔽。文档边界处同块内的跨文档对由掩码处理。
            for q0, q1 in [(b0, b1) for (b0, b1) in blocks if k0 < b1]:
                valid = _block_valid(doc_ids, q0, q1, k0, k1)
                if not torch.any(valid):
                    # 整块被屏蔽：跳过，running 统计量保持不变，不会产生 NaN。
                    continue

                scores = (q[q0:q1] @ k[k0:k1].T) * scale
                scores = torch.where(valid, scores, NEG_INF)

                block_max = torch.amax(scores, dim=1)  # 完全屏蔽行为 -inf
                active = torch.isfinite(block_max)

                safe_prev = torch.where(active, row_max[q0:q1], 0.0)
                safe_block = torch.where(active, block_max, 0.0)
                new_max = torch.maximum(safe_prev, safe_block)
                new_max = torch.where(active, new_max, NEG_INF)

                # 旧累积量的指数修正因子；非活动行保持 0（其输出本就为 0）。
                prev_corr = torch.where(
                    active, torch.exp(safe_prev - new_max), 0.0
                )
                # 块内未归一化权重 exp(z - new_max)；屏蔽位置（含 -inf）为 0。
                block_w = torch.where(
                    active.unsqueeze(1),
                    torch.exp(scores - new_max.unsqueeze(1)),
                    0.0,
                )
                block_denom = torch.sum(block_w, dim=1)

                # out 始终保存“截至当前块的归一化输出”：
                #   numer = O_old * (D_old * exp(m_old - m_new)) + W_block @ V
                #   D_new = D_old * exp(...) + block_denom
                #   O_new = numer / D_new
                # 非活动行分子、分母均为 0，用 tiny 兜底避免 0/0（结果仍为 0）。
                old_scaled_denom = row_denom[q0:q1] * prev_corr
                numer = (
                    out[q0:q1] * old_scaled_denom.unsqueeze(1)
                    + block_w @ v[k0:k1]
                )
                new_denom = old_scaled_denom + block_denom
                row_denom[q0:q1] = new_denom
                out[q0:q1] = numer / new_denom.clamp_min(
                    torch.finfo(torch.float64).tiny
                ).unsqueeze(1)
                row_max[q0:q1] = new_max

        # 每个 query 至少能读取自身，最终分母严格为正；out 已在循环内归一化。

        # 跨前后向只保存 O(N)（外加固定 d 维）张量，不保存任何 N x N 矩阵。
        ctx.save_for_backward(q, k, v, doc_ids, row_max, row_denom, out)
        ctx.scale = float(scale)
        ctx.chunk_size = int(chunk_size)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, doc_ids, row_max, row_denom, out = ctx.saved_tensors
        n, d = q.shape
        scale: float = ctx.scale
        chunk_size: int = ctx.chunk_size
        blocks = _split(n, chunk_size)

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        # dV += P^T dO；dO_i^T O_i = sum_j P_ij dO_i^T V_j，逐块累加。
        do_sum = torch.sum(grad_out * out, dim=1)  # O(N)

        for k0, k1 in blocks:
            for q0, q1 in [(b0, b1) for (b0, b1) in blocks if k0 < b1]:
                valid = _block_valid(doc_ids, q0, q1, k0, k1)
                if not torch.any(valid):
                    continue

                # 用保存的行最大值/分母重算块内概率，块矩阵仅 O(C^2)。
                scores = (q[q0:q1] @ k[k0:k1].T) * scale
                prob = torch.where(
                    valid,
                    torch.exp(scores - row_max[q0:q1, None])
                    / row_denom[q0:q1, None],
                    0.0,
                )

                grad_v[k0:k1] += prob.T @ grad_out[q0:q1]

                dot = grad_out[q0:q1] @ v[k0:k1].T  # dO_i^T V_j
                d_prob = prob * (dot - do_sum[q0:q1, None])
                d_scores = d_prob * scale  # softmax 雅可比 + 链式法则

                grad_q[q0:q1] += d_scores @ k[k0:k1]
                grad_k[k0:k1] += d_scores.T @ q[q0:q1]

        return grad_q, grad_k, grad_v, None, None, None


def chunked_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    doc_ids: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> torch.Tensor:
    """分块文档内因果注意力的函数式入口。"""
    return ChunkedCausalAttention.apply(q, k, v, doc_ids, scale, chunk_size)
