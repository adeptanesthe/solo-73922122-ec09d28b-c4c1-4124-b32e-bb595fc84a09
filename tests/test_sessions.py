"""增量注意力会话核心测试：增量正确性、分叉隔离、修订号并发与非法请求。"""

from __future__ import annotations

import threading

import pytest
import torch

from app.reference import dense_reference
from app.session_schemas import INCREMENTAL_TOLERANCE, MAX_BRANCH_TOKENS
from app.sessions import IncrementalSessionStore, SessionError


def _rand_tokens(n: int, seed: int = 0, scale: float = 7.0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 16, generator=g, dtype=torch.float64) * scale
    k = torch.randn(n, 16, generator=g, dtype=torch.float64) * scale
    v = torch.randn(n, 16, generator=g, dtype=torch.float64) * scale
    # 保证分量在 [-8, 8]
    for t in (q, k, v):
        t.clamp_(-8.0, 8.0)
    return q, k, v


def _append(store, sid, bid, revision, q, k, v, new_document):
    return store.append_token(sid, bid, revision, q.tolist(), k.tolist(), v.tolist(),
                              new_document)


def _build(store, s, doc_lengths, seed=0):
    summary = store.create_session(s)
    sid = summary.session_id
    bid = summary.branches[0].branch_id
    n = sum(doc_lengths)
    q, k, v = _rand_tokens(n, seed)
    outs, responses = [], []
    index = 0
    for doc_id, length in enumerate(doc_lengths):
        for offset in range(length):
            resp = _append(
                store, sid, bid, index, q[index], k[index], v[index], offset == 0
            )
            assert resp.revision == index + 1
            assert resp.document_index == doc_id
            assert resp.document_length == offset + 1
            outs.append(resp.output)
            responses.append(resp)
            index += 1
    doc_ids = torch.cat(
        [torch.full((length,), doc_id, dtype=torch.int64)
         for doc_id, length in enumerate(doc_lengths)]
    )
    return sid, bid, (q, k, v, doc_ids), torch.tensor(outs, dtype=torch.float64), responses


@pytest.mark.parametrize(
    "s,doc_lengths,seed",
    [
        (1.0, [8], 0),
        (0.5, [1, 1], 1),
        (37.7, [3, 1, 9, 2], 2),
        (999.0, [100, 100, 100, 100, 112], 3),  # N=512 且含大量文档边界
        (1000.0, [512], 4),  # 单文档上限 + 大分数
        (2.5, [16, 1], 5),
    ],
)
def test_incremental_outputs_match_dense_reference(s, doc_lengths, seed):
    store = IncrementalSessionStore()
    sid, bid, (q, k, v, doc_ids), got, _ = _build(store, s, doc_lengths, seed)
    expected = dense_reference(q, k, v, doc_ids, s / 4.0)
    assert torch.isfinite(got).all()
    err = float(torch.max(torch.abs(got - expected)).item())
    assert err <= INCREMENTAL_TOLERANCE, f"最大绝对误差 {err:.3e}"

    result = store.verify_branch(sid, bid)
    assert result.passed is True
    assert result.max_abs_err <= INCREMENTAL_TOLERANCE
    assert result.checked_tokens == sum(doc_lengths)


def test_no_cross_document_leakage():
    """新文档首 token 的输出必须等于其自身 V，与历史 K/V 无关。"""
    store = IncrementalSessionStore()
    sid, bid, (q, k, v, _), _, responses = _build(store, 1.0, [4, 3], seed=7)
    first_of_second_doc = torch.tensor(responses[4].output, dtype=torch.float64)
    assert torch.allclose(first_of_second_doc, v[4], atol=1e-12, rtol=0)

    # 反向验证：改变第二篇文档之前的所有 K/V 不得改变该 token 的输出。
    store2 = IncrementalSessionStore()
    summary = store2.create_session(1.0)
    sid2, bid2 = summary.session_id, summary.branches[0].branch_id
    q2, k2, v2 = _rand_tokens(1, seed=99)
    for i in range(3):
        _append(store2, sid2, bid2, i, q2[0], k2[0], v2[0], i == 0)
    resp = _append(store2, sid2, bid2, 3, q[4], k[4], v[4], True)
    assert torch.allclose(
        torch.tensor(resp.output, dtype=torch.float64), v[4], atol=1e-12, rtol=0
    )


def test_history_outputs_not_recomputed():
    """追加只依赖缓存 K/V：追加前后缓存的历史输出逐位一致。"""
    store = IncrementalSessionStore()
    sid, bid, (q, k, v, _), _, responses = _build(store, 3.0, [5, 4], seed=8)
    first = torch.tensor(responses[0].output, dtype=torch.float64)
    for _ in range(3):
        revision = store.get_branch(sid, bid).revision
        _append(store, sid, bid, revision, q[8], k[8], v[8], False)
    assert torch.equal(
        torch.tensor(responses[0].output, dtype=torch.float64), first
    )
    assert store.verify_branch(sid, bid).passed is True


def test_fork_copies_parent_state_and_isolates_children():
    store = IncrementalSessionStore()
    sid, bid, (q, k, v, _), parent_outs, _ = _build(store, 2.0, [6], seed=9)
    parent_revision = store.get_branch(sid, bid).revision

    fork = store.fork(sid, bid, parent_revision)
    child = fork.child.branch_id
    assert fork.parent_revision == parent_revision
    assert fork.child.parent_id == bid
    assert fork.child.fork_revision == parent_revision
    assert fork.child.token_count == parent_revision
    assert fork.child.revision == parent_revision

    # 父支继续追加一个 token；子支新增另一篇文档。
    _append(store, sid, bid, parent_revision, q[0], k[0], v[0], False)
    child_resp = _append(store, sid, child, parent_revision, q[1], k[1], v[1], True)

    # 父支未受子支影响。
    assert store.get_branch(sid, bid).revision == parent_revision + 1
    assert store.get_branch(sid, bid).document_lengths == [7]
    # 子支未受父支影响，其新文档只有自身 token。
    assert store.get_branch(sid, child).revision == parent_revision + 1
    assert store.get_branch(sid, child).document_lengths == [6, 1]
    assert torch.allclose(
        torch.tensor(child_resp.output, dtype=torch.float64), v[1], atol=1e-12, rtol=0
    )

    # 子支全量重算：父支旧 token 的输出不受父支新增影响。
    result = store.verify_branch(sid, child)
    assert result.passed is True
    assert result.checked_tokens == 7
    # 父支与子支的前 6 行必须一致。
    parent_check = store.verify_branch(sid, bid)
    assert parent_check.passed is True


def test_fork_requires_matching_parent_revision():
    store = IncrementalSessionStore()
    sid, bid, (q, k, v, _), _, _ = _build(store, 1.0, [3], seed=10)
    before = store.get_branch(sid, bid)
    with pytest.raises(SessionError) as exc:
        store.fork(sid, bid, before.revision - 1)
    assert exc.value.status_code == 409
    assert store.get_session(sid).branches == [before]


def test_stale_and_invalid_appends_do_not_mutate_branch():
    store = IncrementalSessionStore()
    sid, bid, (q, k, v, _), _, _ = _build(store, 1.0, [2], seed=11)
    before = store.get_branch(sid, bid)

    # 过期修订号
    with pytest.raises(SessionError) as exc:
        _append(store, sid, bid, before.revision - 1, q[0], k[0], v[0], False)
    assert exc.value.status_code == 409
    # 未来修订号
    with pytest.raises(SessionError) as exc:
        _append(store, sid, bid, before.revision + 1, q[0], k[0], v[0], False)
    assert exc.value.status_code == 409
    # 非法分量：超出 [-8, 8]、非有限值、维度错误
    bad_q = q[0].clone()
    bad_q[0] = 8.5
    with pytest.raises(SessionError) as exc:
        _append(store, sid, bid, before.revision, bad_q, k[0], v[0], False)
    assert exc.value.status_code == 422
    nan_k = k[0].clone()
    nan_k[3] = float("nan")
    with pytest.raises(SessionError) as exc:
        _append(store, sid, bid, before.revision, q[0], nan_k, v[0], False)
    assert exc.value.status_code == 422
    with pytest.raises(SessionError) as exc:
        store.append_token(sid, bid, before.revision, [0.0] * 15, k[0].tolist(),
                           v[0].tolist(), False)
    assert exc.value.status_code == 422
    # 新会话的根分支为空：首个 token 未开启新文档则拒绝。
    empty = store.create_session(1.0)
    empty_bid = empty.branches[0].branch_id
    with pytest.raises(SessionError) as exc:
        _append(store, empty.session_id, empty_bid, 0, q[0], k[0], v[0], False)
    assert exc.value.status_code == 422
    assert store.get_branch(empty.session_id, empty_bid).token_count == 0

    assert store.get_branch(sid, bid) == before


def test_same_revision_concurrent_append_accepted_once():
    store = IncrementalSessionStore()
    sid, bid, (q, k, v, _), _, _ = _build(store, 1.0, [2], seed=12)
    revision = store.get_branch(sid, bid).revision

    outcomes: list[str] = []
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        barrier.wait()
        try:
            _append(store, sid, bid, revision, q[0], k[0], v[0], False)
            outcomes.append("ok")
        except SessionError as exc:
            outcomes.append(f"{exc.status_code}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count("ok") == 1
    assert outcomes.count("409") == 7
    branch = store.get_branch(sid, bid)
    assert branch.revision == revision + 1
    assert branch.token_count == revision + 1
    assert store.verify_branch(sid, bid).passed is True


def test_token_limit_and_unknown_ids():
    store = IncrementalSessionStore()
    session = store.create_session(1.0)
    sid, bid = session.session_id, session.branches[0].branch_id
    with pytest.raises(SessionError) as exc:
        store.get_branch("missing", bid)
    assert exc.value.status_code == 404
    with pytest.raises(SessionError) as exc:
        store.get_branch(sid, "missing")
    assert exc.value.status_code == 404
    with pytest.raises(SessionError) as exc:
        store.verify_branch(sid, bid)
    assert exc.value.status_code == 409

    # 上限 512：填满后拒绝继续追加，且状态不变。
    q, k, v = _rand_tokens(1, seed=13)
    for i in range(MAX_BRANCH_TOKENS):
        _append(store, sid, bid, i, q[0], k[0], v[0], i == 0)
    full = store.get_branch(sid, bid)
    assert full.token_count == MAX_BRANCH_TOKENS
    assert full.remaining_tokens == 0
    with pytest.raises(SessionError) as exc:
        _append(store, sid, bid, MAX_BRANCH_TOKENS, q[0], k[0], v[0], False)
    assert exc.value.status_code == 409
    assert store.get_branch(sid, bid) == full
    assert store.verify_branch(sid, bid).passed is True


def test_session_s_is_immutable_and_shared_by_branches():
    store = IncrementalSessionStore()
    summary = store.create_session(250.0)
    assert summary.s == 250.0
    assert summary.scale_used == 62.5
    sid, bid = summary.session_id, summary.branches[0].branch_id
    q, k, v = _rand_tokens(1, seed=14)
    resp = _append(store, sid, bid, 0, q[0], k[0], v[0], True)
    assert resp.s == 250.0 and resp.scale_used == 62.5
    # 请求模型拒绝额外字段（不能临时改 s）
    from app.session_schemas import AppendTokenRequest

    with pytest.raises(Exception):
        AppendTokenRequest(
            expected_revision=1, q=[0.0] * 16, k=[0.0] * 16, v=[0.0] * 16,
            new_document=False, s=2.0,
        )


def test_fork_children_match_dense_reference_independently():
    """分叉后父子各自追加：每支的完整输出都必须与其自身序列的稠密重算一致。"""
    store = IncrementalSessionStore()
    sid, bid, (q, k, v, _), parent_outs, _ = _build(store, 1.5, [4, 3], seed=21)
    fork_revision = store.get_branch(sid, bid).revision
    child = store.fork(sid, bid, fork_revision).child.branch_id

    qx, kx, vx = _rand_tokens(4, seed=22)
    # 父支在最后一篇文档内继续追加两 token。
    _append(store, sid, bid, fork_revision, qx[0], kx[0], vx[0], False)
    _append(store, sid, bid, fork_revision + 1, qx[1], kx[1], vx[1], False)
    # 子支先在同一文档续写一个 token，再开启新文档追加一个 token。
    _append(store, sid, child, fork_revision, qx[2], kx[2], vx[2], False)
    _append(store, sid, child, fork_revision + 1, qx[3], kx[3], vx[3], True)

    n = fork_revision
    parent_seq = (torch.cat([q[:n], qx[:2]]), torch.cat([k[:n], kx[:2]]),
                  torch.cat([v[:n], vx[:2]]))
    child_seq = (torch.cat([q[:n], qx[2:]]), torch.cat([k[:n], kx[2:]]),
                 torch.cat([v[:n], vx[2:]]))
    parent_docs = torch.tensor([0] * 4 + [1] * 5, dtype=torch.int64)
    child_docs = torch.tensor([0] * 4 + [1] * 4 + [2], dtype=torch.int64)

    for branch_id, seq, docs in (
        (bid, parent_seq, parent_docs),
        (child, child_seq, child_docs),
    ):
        result = store.verify_branch(sid, branch_id)
        assert result.passed is True
        expected = dense_reference(*seq, docs, 1.5 / 4.0)
        incremental = torch.tensor(
            [_output_of(store, sid, branch_id, i) for i in range(len(docs))],
            dtype=torch.float64,
        )
        err = float(torch.max(torch.abs(incremental - expected)).item())
        assert err <= INCREMENTAL_TOLERANCE, f"{branch_id} 误差 {err:.3e}"
    # 父支状态没有被分叉或子支改写。
    assert store.get_branch(sid, bid).document_lengths == [4, 5]
    assert store.get_branch(sid, child).document_lengths == [4, 4, 1]
    assert store.get_branch(sid, bid).revision == fork_revision + 2
    assert store.get_branch(sid, child).revision == fork_revision + 2


def _output_of(store, sid, bid, index: int) -> list[float]:
    """从会话内部状态读取某分支已提交 token 的增量输出（测试辅助）。"""
    state = store._sessions[sid]
    return [float(x) for x in state.branches[bid].outputs[index].tolist()]
