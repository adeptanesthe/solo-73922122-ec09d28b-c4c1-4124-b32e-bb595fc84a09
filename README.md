# 分块文档内因果注意力核对 API + 增量注意力会话

本服务包含两组能力（共享同一 CPU + float64 注意力语义）：

1. **分块注意力核对**：把多篇文档拼入同一批次（总长 N ≤ 512），核对一个
   自定义分块注意力 autograd 算子与独立稠密参考实现的前向输出及 Q/K/V 梯度。
2. **进程内增量注意力会话**：创建时固定 `s`，逐 token 追加 16 维 Q/K/V 与
   新文档标记，只用该分支缓存的本篇 K/V 计算新 token 的 float64 输出；
   支持带修订号的乐观并发控制与分支分叉。

## 分块注意力核对（原有能力）

- 每个 token 只能读取**同一篇文档中位置不晚于自己**的 token
- 输出：`O = softmax(s · Q Kᵀ / 4) · V`
- Q/K/V 与上游梯度均为 16 维，float64
- 自定义算子前向、后向**均分块**计算，块长 `chunk_size ∈ [1, 32]`
- 不调用任何框架内置注意力算子，也不依赖内置 softmax 原语（online softmax 手写）
- 工作张量与跨前后向保存量规模 **O(N + C²)**，C 为块长；独立稠密参考不受此限制
- 容差：输出与 Q/K/V 梯度的最大绝对误差均 **≤ 1e-8**，否则 `passed=false` 并给出原因

## 目录结构

```
app/
  attention.py          # 自定义分块 autograd 算子（online softmax 前向 + 分块反向）
  reference.py          # 独立稠密参考（完整 N×N 掩码与注意力矩阵，不受内存约束）
  verification.py       # 数据生成、双路计算与误差核对
  schemas.py            # /verify 请求/响应模型与输入校验
  scenarios.py          # 内置边界情形样例
  sessions.py           # 增量会话：分支、修订号、写时复制、独立全量重算核对
  session_schemas.py    # 会话请求/响应模型、向量校验与常量
  main.py               # FastAPI 入口
tests/                  # pytest：正确性、gradcheck、边界、内存/源码审计、API、会话
Dockerfile
docker-compose.yml
```

## 启动（Docker Compose）

```bash
docker compose up --build
# 服务监听 http://localhost:8000
# Swagger 文档：http://localhost:8000/docs
# OpenAPI：     http://localhost:8000/openapi.json
```

## 本地（无 Docker）运行

```bash
pip install -r requirements.txt   # torch 使用 CPU wheel 源
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest -q                         # 运行自测
```

## 接口

### `POST /verify`

请求体：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `seed` | int | ≥ 0，决定 Q/K/V 与上游梯度的确定性生成 |
| `doc_lengths` | int[] | 正整数列表，总长 1..512 |
| `chunk_size` | int | 1..32 |
| `s` | float | 0 < s ≤ 1000 |

查询参数 `include_tensors`（默认 `true`）：是否回传完整输出/梯度张量。

curl 示例：

```bash
curl -X POST "http://localhost:8000/verify?include_tensors=false" \
  -H "Content-Type: application/json" \
  -d '{"seed":42,"doc_lengths":[7,11,5],"chunk_size":8,"s":1.0}'
```

响应关键字段：`passed`、`tolerance=1e-8`、`errors`（四项最大绝对误差）、
`finite_ok`、`failure_reasons`。核对失败是业务结果而非 HTTP 错误，仍返回 200，
通过 `passed=false` 与 `failure_reasons` 明确表达；参数非法返回 422。

### `GET /scenarios` / `POST /scenarios/run`

列出或运行内置边界情形；`POST /scenarios/run?name=<名称>` 只跑单个样例。

## 内置边界情形（题目要求）

| 名称 | 配置 | 覆盖点 |
| --- | --- | --- |
| `doc_cross_chunk` | 文档 7/11/5，块长 8，s=1 | 文档边界落在块内部，切块不改变归一化 |
| `fully_masked_block` | 单文档 16，块长 4，s=2.5 | 整块被因果屏蔽时安全跳过、无 NaN |
| `large_score` | 文档 33/31，块长 32，s=1000 | 大分数下 online softmax 前后向无溢出 |
| `combined_stress` | 文档 100×4+112（N=512），块长 7，s=999 | 三类情形叠加 + 块长不整除 |
| `minimal_single_token` | 两篇各 1 token，块长 32 | 最小批次与退化分块循环 |

```bash
curl -X POST "http://localhost:8000/scenarios/run"
```

## 增量注意力会话

同一会话内所有分支共享创建时固定的 `s`（请求体不接受其它 s；多余字段返回 422）。
会话只存在于**当前 API 进程内**，进程重启即清空，不做持久化。

### 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/sessions` | 创建会话，`{"s": 1.0}`；返回 `session_id` 与根分支 `branch_id`（201） |
| `GET` | `/sessions/{sid}` | 会话与全部分支摘要 |
| `GET` | `/sessions/{sid}/branches/{bid}` | 单分支摘要（修订号、token 数、文档长度等） |
| `POST` | `/sessions/{sid}/branches/{bid}/tokens` | 追加一个 token，返回该 token 的输出 |
| `POST` | `/sessions/{sid}/branches/{bid}/fork` | 以父支当前修订号分叉 |
| `POST` | `/sessions/{sid}/branches/{bid}/verify` | 该支增量输出 vs 独立全量重算 |

### 追加 token

请求体（拒绝未声明字段）：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `expected_revision` | int | ≥ 0；必须等于该支当前修订号 |
| `q` / `k` / `v` | float[16] | 每个分量有限且落在 `[-8, 8]` |
| `new_document` | bool | 该 token 是否开启新文档；**分支首个 token 必须为 true** |

成功返回：`revision`（新修订号）、`token_index`、`document_index`、
`document_length`、`output`（16 维 float64）。

### 语义与约束

- **文档内因果**：新 token 的分数只与其**当前文档**（缓存末尾连续后缀）的
  K 以及自身 K 计算，`O = softmax(s·QKᵀ/4)·V`；跨文档不可读。
- **只算新 token**：追加仅用该支缓存的 K/V 计算这一行，不重算任何历史输出；
  Q 仅为 `/verify` 的独立全量重算而缓存，增量计算不读历史 Q。
- **首 token**：空分支的首个 token必须是新文档；每个分支最多 512 个 token，
  超出返回 409 且不改动状态。
- **修订号**：`revision` 恒等于该支已提交 token 数；根支从 0 起算，
  分叉子支从父支分叉修订号续号。追加/分叉的 `expected_revision` 不匹配返回 409。
- **并发**：同一会话的追加/分叉由进程内锁串行化；同一修订号的并发追加
  只接受一次，其余收到 409，**且失败请求不改变任何分支**。
- **分叉**：`fork` 携带父支当前修订号；子支复制父支文档结构、Q/K/V 与输出
  （写时复制），`fork_revision` 固定为父支分叉时修订号；父支状态不变，
  父子后续追加互不影响。
- **核对**：`/verify` 用该支全部 Q/K/V 独立做一次稠密全量重算（手工 softmax），
  与该支逐次增量输出比较，误差 ≤ 1e-8 时 `passed=true`。

### 示例

```bash
SID=$(curl -s -X POST localhost:8000/sessions -H 'Content-Type: application/json' \
      -d '{"s":1.0}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
BID=$(curl -s localhost:8000/sessions/$SID | python3 -c \
      'import sys,json;print(json.load(sys.stdin)["branches"][0]["branch_id"])')

# 第 1 个 token：开启文档
curl -s -X POST localhost:8000/sessions/$SID/branches/$BID/tokens \
  -H 'Content-Type: application/json' \
  -d '{"expected_revision":0,"q":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      "k":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      "v":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
      "new_document":true}'

# 分叉、核对
curl -s -X POST localhost:8000/sessions/$SID/branches/$BID/fork \
  -H 'Content-Type: application/json' -d '{"expected_revision":1}'
curl -s -X POST localhost:8000/sessions/$SID/branches/$BID/verify
```

## 实现要点

1. **掩码**：`valid(i,j) = (doc_i == doc_j) ∧ (pos_j ≤ pos_i)`，按块即时构造 C×C 布尔掩码。
2. **前向 online softmax**：逐 key 块维护每行的运行最大值 `m`、归一化分母 `d`
   与归一化输出 `O`。遇到新块时以 `exp(m_old − m_new)` 修正旧累积量，
   全程只做最大值平移，不保存 N×N 矩阵。
3. **整块屏蔽**：块内无有效位置时直接跳过；行级 `-inf` 通过 `isfinite` 判定，
   以 0 参与指数运算，杜绝 `−inf − −inf`、`0/0` 产生的 NaN。
4. **后向**：用保存的行最大值/分母在每个块上**重算**块内概率 P（仅 C×C），
   利用 `dP = P ⊙ (dO·Vᵀ − Σ_j P_ij dO_iᵀV_j)` 与 `dS = scale·dP` 分块累加
   `dQ/dK/dV`；保存量仅为 Q/K/V、doc_ids、行最大值、行分母、输出，均为 O(N)。
5. **数值稳定**：float64 全程；s=1000 时原始分数很大，最大值平移保证
   `exp` 输入 ≤ 0，前后向均为有限值。
6. **确定性**：同一 `seed` 生成完全相同的输入与结果。
7. **增量会话**：不使用掩码矩阵——同文档历史恰为缓存末尾连续后缀，直接切片
   读取；分叉共享张量并在追加时 `torch.cat` 生成新张量，实现写时复制。

## 自测结果

`pytest -q` 共 49 项全部通过（本机 Python 3.11 + torch 2.2.2+cpu）：

- **原有 27 项**：11 组参数化前向+梯度核对（误差 ≤ 1e-8，实测 1e-12 ~ 1e-16）、
  `torch.autograd.gradcheck`、块长不变性、确定性、掩码语义、大分数有限性、
  保存张量线性规模审计、禁用算子源码审计、全部 API 端点与参数校验。
- **新会话核心 15 项**：六组配置（含 N=512、s=1000、跨文档）逐 token 增量输出
  对照独立稠密参考误差 ≤ 1e-8（实测多为 0）；跨文档隔离（新文档首 token 输出
  等于其 V，且与历史 K/V 无关）；历史输出不被改写；分叉隔离与父子各自
  对照稠密参考；过期/超前修订号、非法向量、首 token 未开新文档均 409/422
  且状态不变；8 线程同修订号并发追加仅 1 次成功、7 次 409；512 上限与未知 ID。
- **新会话 HTTP 7 项**：创建/校验/`/docs` 与 OpenAPI 路径、追加输出与
  `/verify`、非法请求不改状态、同修订号重复提交、分叉分支隔离、
  8 并发 HTTP 追加仅 1 次 200、未知分支与空分支核对。
- 另以真实 `uvicorn` 进程做了端到端 curl 冒烟：建会话 → 追加 → 过期 409 →
  `/verify` 通过（实测 1.39e-17）→ 分叉，`/docs` 200。

## 未完成 / 限制

- 会话状态为**单进程内存**，多 worker / 多副本不共享，重启即丢失；无持久化与
  过期回收。并发控制依赖进程内 `threading.RLock`，不适用于多进程部署。
- 分叉时各支共享底层张量（写时复制），但历史缓存不释放；长会话内存随分支数
  线性增长，无淘汰策略。
- 本轮环境无 Docker 守护进程，**未实际执行 `docker compose up --build`**；
  Dockerfile / compose 的改动只在配置层面核对，构建与容器内健康检查未验证。
- 本轮未运行 `ruff` / `mypy` 等静态检查（环境中未安装）。

