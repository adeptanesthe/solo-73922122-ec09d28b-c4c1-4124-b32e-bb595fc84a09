"""FastAPI 入口：分块注意力核对与增量注意力会话服务。

端点
----
* ``GET  /health``                    健康检查
* ``GET  /scenarios``                 内置边界情形样例列表
* ``POST /verify``                    单次核对（自定义分块算子 vs 独立稠密参考）
* ``POST /scenarios/run``             运行全部或单个内置样例
* ``POST /sessions``                  创建增量注意力会话（固定 s）
* ``GET  /sessions/{id}``             会话与全部分支状态
* ``GET  /sessions/{id}/branches/{b}`` 单分支状态
* ``POST /sessions/{id}/branches/{b}/tokens`` 追加一个 token
* ``POST /sessions/{id}/branches/{b}/fork``   以父支当前修订号分叉
* ``POST /sessions/{id}/branches/{b}/verify`` 增量输出 vs 该支全量重算
* ``GET  /docs``                      Swagger UI（自动生成）
"""

from __future__ import annotations

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from .scenarios import SCENARIOS
from .schemas import VerifyRequest, VerifyResponse
from .session_schemas import (
    AppendTokenRequest,
    AppendTokenResponse,
    BranchSummary,
    BranchVerifyResponse,
    ForkRequest,
    ForkResponse,
    SessionCreateRequest,
    SessionSummary,
)
from .sessions import STORE, SessionError
from .verification import TOLERANCE, run_verification

app = FastAPI(
    title="分块因果注意力核对与增量会话 API",
    version="1.1.0",
    description=(
        "CPU + float64 环境下提供两组能力："
        "（1）对自定义分块（文档内因果）注意力 autograd 算子与独立稠密参考实现"
        "进行前向输出和 Q/K/V 梯度核对，容差为 1e-8；"
        "（2）进程内增量注意力会话——创建时固定 s，逐 token 追加 Q/K/V 与"
        "新文档标记，只用该分支缓存的本篇 K/V 计算新 token 的 float64 输出，"
        "支持带修订号的乐观并发控制与分叉。"
    ),
)


@app.exception_handler(SessionError)
def _session_error_handler(request: Request, exc: SessionError) -> JSONResponse:
    """会话业务错误：404/409/422，响应体说明原因且保证状态未被改动。"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "status_code": exc.status_code},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/scenarios")
def list_scenarios() -> dict[str, object]:
    return {
        "tolerance": TOLERANCE,
        "scenarios": [
            {"name": name, "description": s.description, "request": s.request.model_dump()}
            for name, s in SCENARIOS.items()
        ],
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(
    req: VerifyRequest,
    include_tensors: bool = Query(
        default=True, description="是否在响应中回传完整输出/梯度张量"
    ),
) -> VerifyResponse:
    result = run_verification(req, include_tensors=include_tensors)
    # 核对失败属于业务结果而非服务错误：返回 200 并在体内明确 passed=false。
    return result


@app.post("/scenarios/run")
def run_scenarios(
    name: str | None = Query(default=None, description="只运行指定样例；缺省运行全部"),
    include_tensors: bool = Query(default=False),
) -> JSONResponse:
    if name is not None and name not in SCENARIOS:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"未知样例 {name!r}",
                "available": list(SCENARIOS.keys()),
            },
        )
    chosen = [name] if name else list(SCENARIOS.keys())
    results = {}
    all_passed = True
    for key in chosen:
        resp = run_verification(SCENARIOS[key].request, include_tensors=include_tensors)
        all_passed = all_passed and resp.passed
        results[key] = resp.model_dump()
    return JSONResponse(
        content={"all_passed": all_passed, "tolerance": TOLERANCE, "results": results}
    )


# ---------------------------------------------------------------- 增量会话


@app.post("/sessions", response_model=SessionSummary, status_code=201)
def create_session(req: SessionCreateRequest) -> SessionSummary:
    """创建会话：s 在此固定（0 < s ≤ 1000），并自动创建根分支。"""
    return STORE.create_session(req.s)


@app.get("/sessions/{session_id}", response_model=SessionSummary)
def get_session(session_id: str) -> SessionSummary:
    return STORE.get_session(session_id)


@app.get(
    "/sessions/{session_id}/branches/{branch_id}", response_model=BranchSummary
)
def get_branch(session_id: str, branch_id: str) -> BranchSummary:
    return STORE.get_branch(session_id, branch_id)


@app.post(
    "/sessions/{session_id}/branches/{branch_id}/tokens",
    response_model=AppendTokenResponse,
)
def append_token(
    session_id: str, branch_id: str, req: AppendTokenRequest
) -> AppendTokenResponse:
    """追加一个 token，返回它的增量输出。

    - 只用该分支缓存的本篇 K/V 计算新 token，历史输出不重算；
    - ``expected_revision`` 必须等于分支当前修订号，否则 409 且不改动分支；
    - 首个 token 必须 ``new_document=true``；同一修订号的并发追加只接受一次。
    """
    return STORE.append_token(
        session_id=session_id,
        branch_id=branch_id,
        expected_revision=req.expected_revision,
        q=req.q,
        k=req.k,
        v=req.v,
        new_document=req.new_document,
    )


@app.post(
    "/sessions/{session_id}/branches/{branch_id}/fork", response_model=ForkResponse
)
def fork_branch(
    session_id: str, branch_id: str, req: ForkRequest
) -> ForkResponse:
    """以父支当前修订号分叉；父支不变，父子后续追加互不影响。"""
    return STORE.fork(session_id, branch_id, req.expected_revision)


@app.post(
    "/sessions/{session_id}/branches/{branch_id}/verify",
    response_model=BranchVerifyResponse,
)
def verify_branch(session_id: str, branch_id: str) -> BranchVerifyResponse:
    """独立全量重算该分支全部 token，与逐次增量输出比较（容差 1e-8）。"""
    return STORE.verify_branch(session_id, branch_id)
