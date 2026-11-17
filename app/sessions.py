"""进程内增量注意力会话：分支、修订号与逐 token 增量计算。

语义
----
* 一个会话在创建时固定 ``s``，所有分支共用；注意力为
  ``O_i = softmax(s * Q_i K_1..i^T / 4) V_1..i``，但只能读取**同一文档**中
  不晚于自己的 token。
* 追加 token 时只使用该分支缓存的本篇 K/V 计算新 token 的输出：
  新 token 的 softmax 只涉及它自己这一行，历史输出不重算。
* 分支首个 token 必须 ``new_document=true``；跨文档不可读取。
* 分支修订号恒等于该支已提交 token 数，只在该支成功追加时 +1。
  新会话根支从 0 起算，分叉出的子支从父支分叉时的修订号续号。
  追加须携带自身 ``expected_revision``，分叉须携带父支当前修订号；
  同一修订号上的并发追加只接受一次，其余请求收到 409 且不改动任何分支。
* 分叉复制父支的文档结构、Q/K/V 缓存与已提交输出（``fork_revision`` 记录
  父支当时修订号），父支状态不变；父子之后的追加互不影响（缓存写时复制）。
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass

import torch

from .schemas import HEAD_DIM
from .session_schemas import (
    INCREMENTAL_TOLERANCE,
    MAX_BRANCH_TOKENS,
    AppendTokenResponse,
    BranchSummary,
    BranchVerifyResponse,
    ForkResponse,
    SessionSummary,
    check_vector,
)

NEG_INF = float("-inf")


class SessionError(Exception):
    """会话层业务错误；由路由转换为明确的状态码。"""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class BranchState:
    """单个分支的全部状态；不可变字段以分叉时复制的方式隔离。"""

    branch_id: str
    parent_id: str | None
    fork_revision: int | None
    revision: int
    # 已提交 token 的 Q/K/V，形状 (n, 16)，float64；写时复制共享。
    # Q 只用于全量重算核对，增量计算新输出时不读历史 Q。
    queries: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    # 每个 token 的文档编号与文档内位置，形状 (n,)，int64。
    doc_ids: torch.Tensor
    doc_positions: torch.Tensor
    # 文档结构：每篇文档长度；num_documents == len(document_lengths)。
    document_lengths: list[int]
    # 每个 token 在提交时算出的增量输出，供全量重算核对。
    outputs: torch.Tensor
    created_at: float

    @classmethod
    def empty(cls, branch_id: str, parent_id: str | None, fork_revision: int | None,
              created_at: float) -> "BranchState":
        return cls(
            branch_id=branch_id,
            parent_id=parent_id,
            fork_revision=fork_revision,
            revision=0,
            queries=torch.zeros((0, HEAD_DIM), dtype=torch.float64),
            keys=torch.zeros((0, HEAD_DIM), dtype=torch.float64),
            values=torch.zeros((0, HEAD_DIM), dtype=torch.float64),
            doc_ids=torch.zeros((0,), dtype=torch.int64),
            doc_positions=torch.zeros((0,), dtype=torch.int64),
            document_lengths=[],
            outputs=torch.zeros((0, HEAD_DIM), dtype=torch.float64),
            created_at=created_at,
        )

    @property
    def token_count(self) -> int:
        return int(self.doc_ids.numel())

    def summary(self) -> BranchSummary:
        used = self.token_count
        return BranchSummary(
            branch_id=self.branch_id,
            parent_id=self.parent_id,
            fork_revision=self.fork_revision,
            revision=self.revision,
            token_count=used,
            num_documents=len(self.document_lengths),
            document_lengths=list(self.document_lengths),
            max_tokens=MAX_BRANCH_TOKENS,
            remaining_tokens=MAX_BRANCH_TOKENS - used,
            created_at=self.created_at,
        )


@dataclass
class SessionState:
    session_id: str
    s: float
    created_at: float
    branches: dict[str, BranchState]
    lock: threading.RLock


def _append_row(matrix: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    """写时复制追加一行，已有数据不被任何共享它的分支感知。

    返回值总是新张量：父支保留旧张量，子支拿到新张量，二者不再共享存储。
    """
    return torch.cat([matrix, row.reshape(1, -1)], dim=0)


class IncrementalSessionStore:
    """进程内会话表；所有变更都在会话锁内完成。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 会话
    def create_session(self, s: float) -> SessionSummary:
        now = time.time()
        session_id = uuid.uuid4().hex
        root_id = uuid.uuid4().hex
        root = BranchState.empty(root_id, parent_id=None, fork_revision=None,
                                 created_at=now)
        state = SessionState(
            session_id=session_id,
            s=float(s),
            created_at=now,
            branches={root_id: root},
            lock=threading.RLock(),
        )
        with self._lock:
            self._sessions[session_id] = state
        return self._summary(state)

    def _get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise SessionError(404, f"会话 {session_id!r} 不存在")
        return state

    @staticmethod
    def _get_branch(state: SessionState, branch_id: str) -> BranchState:
        branch = state.branches.get(branch_id)
        if branch is None:
            raise SessionError(404, f"分支 {branch_id!r} 不存在于该会话")
        return branch

    def _summary(self, state: SessionState) -> SessionSummary:
        with state.lock:
            branches = [b.summary() for b in state.branches.values()]
        branches.sort(key=lambda b: (b.created_at, b.branch_id))
        return SessionSummary(
            session_id=state.session_id,
            s=state.s,
            scale_used=state.s / 4.0,
            max_tokens_per_branch=MAX_BRANCH_TOKENS,
            branches=branches,
            created_at=state.created_at,
        )

    def get_session(self, session_id: str) -> SessionSummary:
        return self._summary(self._get(session_id))

    def get_branch(self, session_id: str, branch_id: str) -> BranchSummary:
        state = self._get(session_id)
        with state.lock:
            return self._get_branch(state, branch_id).summary()

    # ------------------------------------------------------------ 追加 token
    def append_token(
        self,
        session_id: str,
        branch_id: str,
        expected_revision: int,
        q: list[float],
        k: list[float],
        v: list[float],
        new_document: bool,
    ) -> AppendTokenResponse:
        # 先做纯输入校验（不触碰任何状态），再进入会话锁。
        try:
            q = check_vector(q, "q")
            k = check_vector(k, "k")
            v = check_vector(v, "v")
        except ValueError as exc:
            raise SessionError(422, f"{exc}；请求未生效") from exc

        state = self._get(session_id)
        # 同一会话的追加/分叉全部串行化（进程内锁）：并发携带同一
        # expected_revision 的请求中，只有第一个能通过下面的比较，其余收到 409。
        with state.lock:
            branch = self._get_branch(state, branch_id)
            if expected_revision != branch.revision:
                raise SessionError(
                    409,
                    f"修订号不匹配：expected_revision={expected_revision}，"
                    f"当前修订号为 {branch.revision}；请求未生效",
                )
            used = branch.token_count
            if used >= MAX_BRANCH_TOKENS:
                raise SessionError(
                    409,
                    f"分支已达 {MAX_BRANCH_TOKENS} token 上限，无法继续追加；"
                    "请求未生效",
                )
            if used == 0 and not new_document:
                raise SessionError(
                    422,
                    "分支的首个 token 必须开启新文档（new_document=true）；"
                    "请求未生效",
                )

            q_t = torch.tensor(q, dtype=torch.float64)
            k_t = torch.tensor(k, dtype=torch.float64)
            v_t = torch.tensor(v, dtype=torch.float64)
            scale = state.s / 4.0

            # 新 token 的文档归属：同文档历史 token 即缓存末尾的连续后缀，
            # 因此取“最后 doc_length 个 token”即可，无需任何掩码。
            if new_document:
                document_index = len(branch.document_lengths)
                doc_position = 0
                relevance = 0  # 新文档中只能看到自己
            else:
                document_index = len(branch.document_lengths) - 1
                doc_position = branch.document_lengths[document_index]
                relevance = doc_position
            start = used - relevance

            # ---- 仅用该分支缓存的本篇 K/V 计算这一行 ----
            self_score = float((q_t @ k_t) * scale)
            if relevance:
                # 新 token 对同文档历史 K 的得分：Q_new · K_j，只读缓存 K/V。
                history_scores = (q_t @ branch.keys[start:used].T) * scale
                max_score = max(float(torch.max(history_scores).item()), self_score)
                weights = torch.exp(history_scores - max_score)
                self_weight = math.exp(self_score - max_score)
                denom = float(torch.sum(weights).item()) + self_weight
                numerator = weights @ branch.values[start:used] + self_weight * v_t
            else:
                max_score = self_score
                denom = 1.0
                numerator = v_t.clone()
            out = numerator / denom

            token_index = used

            # ---- 先在局部变量算好全部新状态，再一次性写回分支 ----
            new_queries = _append_row(branch.queries, q_t)
            new_keys = _append_row(branch.keys, k_t)
            new_values = _append_row(branch.values, v_t)
            new_doc_ids = torch.cat(
                [branch.doc_ids, torch.tensor([document_index], dtype=torch.int64)]
            )
            new_doc_positions = torch.cat(
                [branch.doc_positions, torch.tensor([doc_position], dtype=torch.int64)]
            )
            new_outputs = _append_row(branch.outputs, out.detach())
            new_lengths = list(branch.document_lengths)
            if new_document:
                new_lengths.append(1)
            else:
                new_lengths[document_index] += 1

            # 赋值均不会失败，保证分支状态要么完全未变、要么整体提交。
            branch.queries = new_queries
            branch.keys = new_keys
            branch.values = new_values
            branch.doc_ids = new_doc_ids
            branch.doc_positions = new_doc_positions
            branch.outputs = new_outputs
            branch.document_lengths = new_lengths
            branch.revision += 1

            return AppendTokenResponse(
                session_id=session_id,
                branch_id=branch_id,
                previous_revision=expected_revision,
                revision=branch.revision,
                token_index=token_index,
                document_index=document_index,
                document_length=branch.document_lengths[document_index],
                num_documents=len(branch.document_lengths),
                total_tokens=branch.token_count,
                s=state.s,
                scale_used=scale,
                output=[float(x) for x in out.tolist()],
            )

    # ---------------------------------------------------------------- 分叉
    def fork(
        self, session_id: str, parent_branch_id: str, expected_revision: int
    ) -> ForkResponse:
        state = self._get(session_id)
        now = time.time()
        with state.lock:
            parent = self._get_branch(state, parent_branch_id)
            if expected_revision != parent.revision:
                raise SessionError(
                    409,
                    f"父支修订号不匹配：expected_revision={expected_revision}，"
                    f"父支当前修订号为 {parent.revision}；请求未生效",
                )
            child_id = uuid.uuid4().hex
            child = BranchState(
                branch_id=child_id,
                parent_id=parent.branch_id,
                fork_revision=parent.revision,
                # 不变量：任何分支的 revision 恒等于其已提交 token 数。
                # 子支复制父支在 fork_revision 的内容，故从该修订号续号；
                # 此后父子的追加各自递增、互不影响。
                revision=parent.revision,
                # 张量在分支间共享，追加时统一写时复制，互不影响。
                queries=parent.queries,
                keys=parent.keys,
                values=parent.values,
                doc_ids=parent.doc_ids,
                doc_positions=parent.doc_positions,
                document_lengths=list(parent.document_lengths),
                outputs=parent.outputs,
                created_at=now,
            )
            state.branches[child_id] = child
            return ForkResponse(
                parent_branch_id=parent.branch_id,
                parent_revision=parent.revision,
                child=child.summary(),
            )

    # ------------------------------------------------- 全量重算核对（只读）
    def verify_branch(
        self, session_id: str, branch_id: str
    ) -> BranchVerifyResponse:
        state = self._get(session_id)
        with state.lock:
            branch = self._get_branch(state, branch_id)
            if branch.token_count == 0:
                raise SessionError(409, "分支尚无 token，无法核对；请求未生效")
            queries = branch.queries.clone()
            keys = branch.keys.clone()
            values = branch.values.clone()
            doc_ids = branch.doc_ids.clone()
            doc_positions = branch.doc_positions.clone()
            cached = branch.outputs.clone()
            revision = branch.revision

        # 全量稠密重算（与增量路径独立），只比较缓存输出。
        n = queries.shape[0]
        valid = (doc_ids[:, None] == doc_ids[None, :]) & (
            doc_positions[None, :] <= doc_positions[:, None]
        )
        # 手工 softmax（最大值平移），不使用 torch.softmax。
        scores = (queries @ keys.T) * (state.s / 4.0)
        scores = torch.where(
            valid, scores, torch.full_like(scores, NEG_INF)
        )
        row_max = torch.amax(scores, dim=1, keepdim=True)
        shifted = torch.exp(scores - row_max)
        probs = torch.where(valid, shifted, torch.zeros_like(shifted))
        probs = probs / torch.sum(probs, dim=1, keepdim=True)
        reference = probs @ values

        err = float(torch.max(torch.abs(cached - reference)).item())
        finite_ok = bool(
            torch.isfinite(cached).all().item()
            and torch.isfinite(reference).all().item()
        )
        reasons: list[str] = []
        if not (err <= INCREMENTAL_TOLERANCE):
            reasons.append(
                f"增量输出与全量重算的最大绝对误差 {err:.3e} "
                f"超过容差 {INCREMENTAL_TOLERANCE:.0e}"
            )
        if not finite_ok:
            reasons.append("增量或重算输出中存在 NaN/Inf")
        return BranchVerifyResponse(
            branch_id=branch_id,
            revision=revision,
            passed=not reasons,
            tolerance=INCREMENTAL_TOLERANCE,
            max_abs_err=err,
            checked_tokens=n,
            finite_ok=finite_ok,
            failure_reasons=reasons,
        )


STORE = IncrementalSessionStore()
