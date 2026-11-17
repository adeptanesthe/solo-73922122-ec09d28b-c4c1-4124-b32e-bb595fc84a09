"""增量会话 HTTP 端点测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _vec(value: float) -> list[float]:
    return [value] * 16


def _create(s: float = 1.0) -> tuple[str, str]:
    resp = client.post("/sessions", json={"s": s})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return body["session_id"], body["branches"][0]["branch_id"]


def test_session_create_and_docs():
    resp = client.post("/sessions", json={"s": 2.5})
    assert resp.status_code == 201
    body = resp.json()
    assert body["s"] == 2.5
    assert body["scale_used"] == 0.625
    assert body["max_tokens_per_branch"] == 512
    assert len(body["branches"]) == 1
    root = body["branches"][0]
    assert root["revision"] == 0 and root["parent_id"] is None
    assert client.get(f"/sessions/{body['session_id']}").status_code == 200
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/sessions" in paths
    assert "/sessions/{session_id}/branches/{branch_id}/tokens" in paths
    assert "/sessions/{session_id}/branches/{branch_id}/fork" in paths
    assert "/verify" in paths


def test_session_validation():
    for bad in (0.0, -1.0, 1000.1):
        assert client.post("/sessions", json={"s": bad}).status_code == 422
    assert client.post("/sessions", json={"s": 1.0, "extra": 1}).status_code == 422
    assert client.get("/sessions/nope").status_code == 404


def test_append_returns_incremental_output_and_verify_passes():
    sid, bid = _create(1000.0)
    resp = client.post(
        f"/sessions/{sid}/branches/{bid}/tokens",
        json={"expected_revision": 0, "q": _vec(1.0), "k": _vec(2.0),
              "v": _vec(0.5), "new_document": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 1
    assert body["token_index"] == 0
    assert body["document_index"] == 0
    assert body["output"] == [0.5] * 16
    assert body["scale_used"] == 250.0

    resp = client.post(
        f"/sessions/{sid}/branches/{bid}/tokens",
        json={"expected_revision": 1, "q": _vec(1.0), "k": _vec(2.0),
              "v": _vec(0.0), "new_document": False},
    )
    assert resp.status_code == 200
    assert resp.json()["revision"] == 2

    check = client.post(f"/sessions/{sid}/branches/{bid}/verify")
    assert check.status_code == 200
    body = check.json()
    assert body["passed"] is True
    assert body["max_abs_err"] <= 1e-8
    assert body["checked_tokens"] == 2


def test_append_rejects_bad_payloads_without_mutation():
    sid, bid = _create()
    url = f"/sessions/{sid}/branches/{bid}/tokens"
    base = {"expected_revision": 0, "q": _vec(0.0), "k": _vec(0.0),
            "v": _vec(0.0), "new_document": True}

    # 首 token 未开启新文档
    assert client.post(url, json={**base, "new_document": False}).status_code == 422
    # 分量超界
    assert client.post(url, json={**base, "q": _vec(8.01)}).status_code == 422
    # 维度错误
    assert client.post(url, json={**base, "k": [0.0] * 15}).status_code == 422
    # 修订号过期
    assert client.post(url, json={**base, "expected_revision": 3}).status_code == 409

    branch = client.get(f"/sessions/{sid}/branches/{bid}").json()
    assert branch["revision"] == 0 and branch["token_count"] == 0

    assert client.post(url, json=base).status_code == 200
    # 用同一个（已过期的）修订号重复提交 -> 409，状态不变
    assert client.post(url, json=base).status_code == 409
    branch = client.get(f"/sessions/{sid}/branches/{bid}").json()
    assert branch["revision"] == 1 and branch["token_count"] == 1


def test_fork_endpoint_partitions_branches():
    sid, bid = _create(4.0)
    for revision in range(3):
        resp = client.post(
            f"/sessions/{sid}/branches/{bid}/tokens",
            json={"expected_revision": revision, "q": _vec(0.1),
                  "k": _vec(0.2), "v": _vec(0.3),
                  "new_document": revision == 0},
        )
        assert resp.status_code == 200

    stale = client.post(
        f"/sessions/{sid}/branches/{bid}/fork", json={"expected_revision": 2}
    )
    assert stale.status_code == 409
    fork = client.post(
        f"/sessions/{sid}/branches/{bid}/fork", json={"expected_revision": 3}
    )
    assert fork.status_code == 200
    child = fork.json()["child"]
    assert child["parent_id"] == bid
    assert child["fork_revision"] == 3
    assert child["revision"] == 3
    assert child["token_count"] == 3

    # 父子各自追加互不影响
    assert client.post(
        f"/sessions/{sid}/branches/{bid}/tokens",
        json={"expected_revision": 3, "q": _vec(0.4), "k": _vec(0.5),
              "v": _vec(0.6), "new_document": False},
    ).status_code == 200
    child_url = f"/sessions/{sid}/branches/{child['branch_id']}/tokens"
    assert client.post(
        child_url,
        json={"expected_revision": 3, "q": _vec(0.7), "k": _vec(0.8),
              "v": _vec(0.9), "new_document": True},
    ).status_code == 200

    parent_summary = client.get(f"/sessions/{sid}/branches/{bid}").json()
    child_summary = client.get(
        f"/sessions/{sid}/branches/{child['branch_id']}"
    ).json()
    assert parent_summary["document_lengths"] == [4]
    assert child_summary["document_lengths"] == [3, 1]

    for branch_id in (bid, child["branch_id"]):
        assert client.post(
            f"/sessions/{sid}/branches/{branch_id}/verify"
        ).json()["passed"] is True
    session = client.get(f"/sessions/{sid}").json()
    assert len(session["branches"]) == 2


def test_same_revision_parallel_http_appends_accept_one():
    sid, bid = _create()
    url = f"/sessions/{sid}/branches/{bid}/tokens"
    payload = {"expected_revision": 0, "q": _vec(0.1), "k": _vec(0.2),
               "v": _vec(0.3), "new_document": True}

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(lambda _: client.post(url, json=payload).status_code,
                              range(8)))

    assert codes.count(200) == 1
    assert codes.count(409) == 7
    branch = client.get(f"/sessions/{sid}/branches/{bid}").json()
    assert branch["revision"] == 1 and branch["token_count"] == 1


def test_unknown_branch_and_empty_verify():
    sid, _ = _create()
    assert client.get(f"/sessions/{sid}/branches/nope").status_code == 404
    assert client.post(f"/sessions/{sid}/branches/nope/verify").status_code == 404
    root = client.get(f"/sessions/{sid}").json()["branches"][0]["branch_id"]
    assert client.post(f"/sessions/{sid}/branches/{root}/verify").status_code == 409
    assert client.post(
        f"/sessions/{sid}/branches/{root}/fork", json={"expected_revision": 0}
    ).status_code == 200
