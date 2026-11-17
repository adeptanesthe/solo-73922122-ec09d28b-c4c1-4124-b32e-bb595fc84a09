"""FastAPI 端点测试（使用 TestClient，无需真实网络端口）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.scenarios import SCENARIOS

client = TestClient(app)


def test_health_and_docs():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    openapi = client.get("/openapi.json").json()
    assert "/verify" in openapi["paths"]


def test_scenarios_listed():
    data = client.get("/scenarios").json()
    assert set(data["scenarios"] and [s["name"] for s in data["scenarios"]]) == set(
        SCENARIOS.keys()
    )


def test_verify_basic_pass():
    resp = client.post(
        "/verify",
        json={"seed": 42, "doc_lengths": [7, 11, 5], "chunk_size": 8, "s": 1.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert body["failure_reasons"] == []
    assert body["finite_ok"] is True
    assert body["errors"]["output_max_abs_err"] <= 1e-8
    assert len(body["output"]) == 23
    assert len(body["output"][0]) == 16


def test_verify_without_tensors():
    resp = client.post(
        "/verify?include_tensors=false",
        json={"seed": 1, "doc_lengths": [4, 4], "chunk_size": 4, "s": 2.0},
    )
    body = resp.json()
    assert body["passed"] is True
    assert body["output"] is None


def test_all_builtin_scenarios_pass():
    resp = client.post("/scenarios/run?include_tensors=false")
    assert resp.status_code == 200
    body = resp.json()
    assert body["all_passed"] is True
    for name, result in body["results"].items():
        assert result["passed"] is True, (name, result["failure_reasons"])
        assert result["finite_ok"] is True


def test_run_single_scenario():
    resp = client.post("/scenarios/run?name=large_score")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["results"].keys()) == {"large_score"}
    assert body["results"]["large_score"]["passed"] is True


def test_run_unknown_scenario_404():
    resp = client.post("/scenarios/run?name=nope")
    assert resp.status_code == 404


def test_validation_errors():
    # 总长超限
    resp = client.post(
        "/verify",
        json={"seed": 0, "doc_lengths": [300, 300], "chunk_size": 8, "s": 1.0},
    )
    assert resp.status_code == 422

    # 非正长度
    resp = client.post(
        "/verify",
        json={"seed": 0, "doc_lengths": [0], "chunk_size": 8, "s": 1.0},
    )
    assert resp.status_code == 422

    # 块长越界
    resp = client.post(
        "/verify",
        json={"seed": 0, "doc_lengths": [8], "chunk_size": 33, "s": 1.0},
    )
    assert resp.status_code == 422

    # s 越界（0 与 >1000）
    for bad_s in (0.0, 1000.1):
        resp = client.post(
            "/verify",
            json={"seed": 0, "doc_lengths": [8], "chunk_size": 8, "s": bad_s},
        )
        assert resp.status_code == 422
