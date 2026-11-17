"""题目要求的边界情形调用样例（可通过 API 直接复跑）。"""

from __future__ import annotations

from .schemas import ScenarioSummary, VerifyRequest

SCENARIOS: dict[str, ScenarioSummary] = {
    "doc_cross_chunk": ScenarioSummary(
        name="doc_cross_chunk",
        description=(
            "三篇文档（长度 7/11/5，共 23 token），块长 8：文档边界落在块内部，"
            "核对切块不改变文档内因果归一化。"
        ),
        request=VerifyRequest(seed=42, doc_lengths=[7, 11, 5], chunk_size=8, s=1.0),
    ),
    "fully_masked_block": ScenarioSummary(
        name="fully_masked_block",
        description=(
            "单篇 16 token、块长 4：第 1 个 query 块对第 3/4 个 key 块整块被因果屏蔽，"
            "核对整块屏蔽被安全跳过且不产生 NaN。"
        ),
        request=VerifyRequest(seed=7, doc_lengths=[16], chunk_size=4, s=2.5),
    ),
    "large_score": ScenarioSummary(
        name="large_score",
        description=(
            "两篇文档（长度 33/31）、块长 32、s=1000：原始分数极大，"
            "核对 online softmax 的最大值平移在前后向均无溢出/NaN。"
        ),
        request=VerifyRequest(seed=123, doc_lengths=[33, 31], chunk_size=32, s=1000.0),
    ),
    "combined_stress": ScenarioSummary(
        name="combined_stress",
        description=(
            "五篇文档（长度 100/100/100/100/112，共 512 token，达到总长上限），"
            "块长 7（不能整除），s=999：同时覆盖跨块文档边界、整块屏蔽与大分数。"
        ),
        request=VerifyRequest(
            seed=2026,
            doc_lengths=[100, 100, 100, 100, 112],
            chunk_size=7,
            s=999.0,
        ),
    ),
    "minimal_single_token": ScenarioSummary(
        name="minimal_single_token",
        description=(
            "两篇各 1 token 的最小批次，块长 32（块数大于所需）："
            "每个 query 只能读取自身，核对退化情形与分块循环边界。"
        ),
        request=VerifyRequest(seed=1, doc_lengths=[1, 1], chunk_size=32, s=0.5),
    ),
}
