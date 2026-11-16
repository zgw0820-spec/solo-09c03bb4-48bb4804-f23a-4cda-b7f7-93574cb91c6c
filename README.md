# 集合降雨超阈值概率 API

防汛预报员把**同一预报时次**的 2～8 个集合成员降雨网格，合成为逐格的**超阈值概率**产品，
供值班员圈定高风险区域；事后再用**实测雨量**对当时发布的概率做逐时次 Brier 核验。
技术栈：**FastAPI + SQLite（WAL）+ NumPy**，可双实例部署、共享同一数据库文件。

## 1. 业务规则与实现对照

| 需求 | 实现方式 |
| --- | --- |
| 登记 UTC 时次、统一网格尺寸、2～8 个成员、雨量阈值 | `POST /groups`，校验时区（必须带时区，统一转 UTC）、成员数 2–8、阈值非负且唯一、网格上限 |
| 逐成员提交非负雨量矩阵、缺测格、递增修订号 | `PUT /groups/{gid}/members/{mid}`，body 含 `rain`、可选 `missing`（0/1 网格，1=缺测）、`revision`（≥1 且严格递增） |
| 同版同文重送不重复 | 同 `revision` 且内容 SHA-256 相同 → 返回 `duplicate=true`，不写新行、不影响产品 |
| 异文拒收 | 同 `revision` 但内容不同 → `409 REVISION_CONTENT_CONFLICT` |
| 成员齐全才可发布 | 任一成员无修订 → `409 MEMBERS_INCOMPLETE` |
| 逐格概率 | `p = 有效成员中 rain ≥ 阈值 的数量 / 该格有效成员数`（NumPy 向量化，按阈值分别计算） |
| 返回概率、分母与所用修订 | 产品含每阈值的 `probability`、`denominator` 矩阵及 `members_used`（成员→修订号） |
| 全缺测格返回空值 | 分母 `0`，概率序列化为 JSON `null` |
| 更正成员后旧产品立即过期 | 新修订入库的同一事务里把该组所有产品置 `is_current=0`；历史行保留 |
| 历史仍可查 | `GET /groups/{gid}/products` 与 `.../products/{id}` 可查已过期产品，响应中 `is_current=false` |
| 当前查询不能把旧概率作为预警依据 | `GET .../products/current` 无现行产品时返回 `404 NO_CURRENT_PRODUCT`，绝不返回过期数据 |
| 两实例交错上传/发布只能产生同组修订的完整产品 | 发布在写事务内重新核对“每个成员的最新修订”，若计算期间发生更正则整体重试；产品只以一致修订集落库（见下节） |
| 计算失败不得留半份当前结果 | 产品主行与全部阈值行在同一 `BEGIN IMMEDIATE` 事务插入，任何异常整体回滚（自测含故障注入验证） |
| 慢计算不阻塞其他时次入仓 | NumPy 计算在写事务**之外**完成，计算期间不持写锁；仅在最后落库阶段短时加锁 |
| 可启动的双实例 Docker Compose 和 Swagger | `docker compose up` 起 `api-a`(:8081)/`api-b`(:8082)，共享命名卷；Swagger 在 `/docs` |
| 首次实测锁定当时现行概率产品 | `PUT /groups/{gid}/observations` 首次接纳与网格同形的非负实测雨量（含缺测格、修订号）时，在同一写事务内锁定当时 `is_current=1` 的产品；当时无现行产品则 `404 NO_CURRENT_PRODUCT` 拒收。锁存入 `observations.locked_product_id`，成员后续更正**不改变**锁定对象 |
| 实测修订递增、同版同文幂等、异文拒收 | 与成员提交同一套规则：低修订号 `409 OBSERVATION_REVISION_CONFLICT`；同修订号同 SHA-256 返回 `duplicate=true`；同修订号异文 `409 OBSERVATION_REVISION_CONTENT_CONFLICT` |
| 更正实测旧核验立即失效 | 新实测修订入库的同一事务里把该组全部核验置 `is_current=0`（历史保留）；重算仍评估**锁定产品**而非最新产品 |
| 逐阈值 Brier 均方误差 | 每个阈值只对「实测有效（缺测格=0）**且**预测概率非空（产品分母>0）」的格点计算 `mean((p - 1{rain≥threshold})²)`，返回每阈值 `brier_score` 与 `valid_cells`；零有效格时分数为 JSON `null` |
| 现行核验对应最新实测修订与锁定产品 | `POST /groups/{gid}/verifications` 按最新实测修订计算并落库；`GET .../verifications/current` 无现行核验返回 `404 NO_CURRENT_VERIFICATION`，绝不返回已失效行 |
| 核验计算失败不留半份现行结果 | 核验主行与全部阈值行在同一 `BEGIN IMMEDIATE` 事务插入，异常整体回滚（自测含故障注入验证） |
| 慢核验不阻塞其他时次成员入仓 | 与发布同构：快照读后在事务外做 NumPy 计算，落库前在写事务内复核实测修订，期间被更正则整体重试（最多 3 次） |
| 核验历史可查 | `GET /groups/{gid}/verifications` 与 `.../verifications/{id}` 可查已失效核验，响应中 `is_current=false`，且每行都带 `locked_product_id` |

### 交错发布的一致性（核心并发设计）

1. 先在一个只读事务里取快照：每个成员的**最新修订号**及对应网格；
2. 释放锁，在事务外做 NumPy 概率计算（慢计算不阻塞任何写入）；
3. 重新进入 `BEGIN IMMEDIATE` 写事务，再次读取每个成员最新修订号：
   - 与快照一致 → 旧产品批量过期、新产品与全部阈值行原子写入；
   - 不一致（计算期间有更正）→ 回滚并按新修订集整体重试（最多 3 次）；
4. 同修订集重复发布走唯一约束/幂等路径，两个实例并发发布同一版本只返回同一个产品。

SQLite 启用 WAL + `busy_timeout=10s`；所有写事务为 `BEGIN IMMEDIATE`，进程间靠单写者锁串行化，
锁竞争失败返回 `409/503` 提示重试，不会产生脏写或半份产品。

### 实测核验的锁定与一致性（新增核心设计）

1. **首接纳即锁定**：首个实测修订在 `BEGIN IMMEDIATE` 事务内读取当时 `is_current=1` 的产品并把
   `product_id` 一并写入 `observations` 行；无现行产品则整体回滚并拒收。锁定只发生一次，之后每个
   实测修订都复用同一个 `locked_product_id`——成员再怎么更正产品，核验对象都不被改写。
2. **核验流程与发布同构**：只读事务取「最新实测修订 + 锁定产品阈值行」快照 → 事务外做 NumPy
   Brier 计算（不持写锁，慢核验不阻塞任何成员/实测入仓）→ 重新进入 `BEGIN IMMEDIATE` 复核实测
   最新修订号：一致则旧核验批量失效、新核验与全部阈值行原子写入；不一致（计算期间实测被更正）
   则回滚并按新修订整体重试（最多 3 次）。
3. **同修订重复核验幂等**：`verifications` 上有 `UNIQUE(group_id, obs_revision)`，并发/重复核验
   只返回同一行；唯一约束冲突时回滚并读取已提交的获胜行。
4. **实测更正即失效**：实测新修订入库的同一事务把旧核验置 `is_current=0`，因此「现行核验」始终
   对应最新实测修订；历史核验保留可查，且永远记录当时锁定的产品 ID，便于事后追溯核验对象。

## 2. API 一览（Swagger 见 `/docs`，OpenAPI 见 `/openapi.json`）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/groups` | 登记时次/网格/成员/阈值 |
| GET | `/groups` | 列出时次 |
| GET | `/groups/{gid}` | 时次状态（各成员最新修订、是否齐全、当前产品 ID） |
| PUT | `/groups/{gid}/members/{mid}` | 提交/更正成员网格修订 |
| POST | `/groups/{gid}/publish` | 发布概率产品（成员齐全） |
| GET | `/groups/{gid}/products/current` | 当前可用于预警的产品（无则 404） |
| GET | `/groups/{gid}/products` | 全部产品（含过期历史） |
| GET | `/groups/{gid}/products/{pid}` | 按 ID 查历史产品 |
| PUT | `/groups/{gid}/observations` | 提交/更正事后实测雨量（首次提交锁定现行产品） |
| POST | `/groups/{gid}/verifications` | 对锁定产品执行逐阈值 Brier 核验（幂等） |
| GET | `/groups/{gid}/verifications/current` | 当前核验（对应最新实测修订，无则 404） |
| GET | `/groups/{gid}/verifications` | 全部核验（含失效历史） |
| GET | `/groups/{gid}/verifications/{vid}` | 按 ID 查历史核验 |

### 示例

```bash
# 1) 登记：2x2 网格、3 个成员、两个阈值
curl -s localhost:8081/groups -H 'Content-Type: application/json' -d '{
  "forecast_time": "2026-11-16T00:00:00Z",
  "rows": 2, "cols": 2,
  "member_names": ["m1", "m2", "m3"],
  "thresholds": [5.0, 10.0]
}'

# 2) 逐成员提交（m3 第二行两个格缺测）
curl -s -X PUT localhost:8081/groups/$GID/members/m1 -H 'Content-Type: application/json' \
  -d '{"rain":[[0,10],[20,0]],"revision":1}'
curl -s -X PUT localhost:8082/groups/$GID/members/m2 -H 'Content-Type: application/json' \
  -d '{"rain":[[10,10],[0,0]],"revision":1}'
curl -s -X PUT localhost:8081/groups/$GID/members/m3 -H 'Content-Type: application/json' \
  -d '{"rain":[[5,0],[99,0]],"missing":[[0,0],[1,1]],"revision":1}'

# 3) 发布并查询
curl -s -X POST localhost:8081/groups/$GID/publish
curl -s localhost:8082/groups/$GID/products/current

# 4) 事后实测：首次提交即锁定当前产品（无现行产品会被拒收）
curl -s -X PUT localhost:8081/groups/$GID/observations -H 'Content-Type: application/json' \
  -d '{"rain":[[0,10],[20,0]],"revision":1}'
# -> {"revision":1,"duplicate":false,"locked_product_id":1,...}

# 5) 逐阈值 Brier 核验（重复调用幂等，返回同一核验 ID）
curl -s -X POST localhost:8082/groups/$GID/verifications
# 6) 实测更正：旧核验立即失效，但重算仍评估 locked_product_id 指向的旧产品
curl -s -X PUT localhost:8081/groups/$GID/observations -H 'Content-Type: application/json' \
  -d '{"rain":[[1,1],[1,1]],"revision":2}'
curl -s -X POST localhost:8081/groups/$GID/verifications
curl -s localhost:8082/groups/$GID/verifications   # 历史含失效行
```

产品响应片段（全缺测格 `null` + 分母 0）：

```json
{
  "product_id": 1,
  "is_current": true,
  "members_used": [
    {"member_idx": 0, "member_id": "m1", "revision": 1},
    {"member_idx": 1, "member_id": "m2", "revision": 1},
    {"member_idx": 2, "member_id": "m3", "revision": 1}
  ],
  "thresholds": [
    {
      "threshold": 5.0,
      "probability": [[0.6666666666666666, 0.6666666666666666], [0.5, 0.0]],
      "denominator": [[3, 3], [2, 2]]
    }
  ]
}
```

核验响应片段（零有效格的阈值 `brier_score` 为 `null`、`valid_cells` 为 0）：

```json
{
  "verification_id": 1,
  "obs_revision": 1,
  "locked_product_id": 1,
  "is_current": true,
  "thresholds": [
    {"threshold": 5.0, "brier_score": 0.2013888888888889, "valid_cells": 4},
    {"threshold": 10.0, "brier_score": 0.11805555555555555, "valid_cells": 4}
  ]
}
```

错误统一为 `{"error": {"code": "...", "message": "..."}}`：
`VALIDATION_ERROR(422)`、`GRID_MISMATCH(422)`、`NEGATIVE_RAIN(422)`、
`REVISION_CONFLICT(409)`、`REVISION_CONTENT_CONFLICT(409)`、`MEMBERS_INCOMPLETE(409)`、
`NO_CURRENT_PRODUCT(404)`、`GROUP_NOT_FOUND/MEMBER_NOT_FOUND/PRODUCT_NOT_FOUND(404)`、
`OBSERVATION_REVISION_CONFLICT(409)`、`OBSERVATION_REVISION_CONTENT_CONFLICT(409)`、
`NO_OBSERVATION(409)`、`NO_CURRENT_VERIFICATION(404)`、`VERIFICATION_NOT_FOUND(404)`、
`REVISION_CHANGED(409)`、`COMPUTE_FAILED(500)`、`DATABASE_BUSY(503)`。

## 3. 运行

### Docker Compose（双实例）

```bash
docker compose up --build
# api-a: http://localhost:8081   api-b: http://localhost:8082
# Swagger: http://localhost:8081/docs 或 http://localhost:8082/docs
# SQLite 文件在命名卷 rain-data 中，两实例共享（WAL 模式）
```

### 本地 Python（无 Docker）

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./run_local.sh                 # 起两个 uvicorn：8081 / 8082，共享 ./data/rainfall.db
```

环境变量：`RAIN_DB_PATH`（默认 `/data/rainfall.db`）、`RAIN_BUSY_TIMEOUT_MS`（默认 10000）。
另有四个**默认关闭**的测试钩子：`RAIN_TEST_PUBLISH_DELAY_S`（发布计算延时）、
`RAIN_TEST_FAIL_MARKER`（该文件存在时发布计算确定性失败）、
`RAIN_TEST_VERIFY_DELAY_S`（核验计算延时）、`RAIN_TEST_VERIFY_FAIL_MARKER`（该文件存在时核验确定性失败），
仅用于自测，生产不要设置。

## 4. 自测

`tests/selftest.py` 只用标准库，默认自动拉起两个共享同一 SQLite 的 uvicorn 实例并跑 113 项断言：

```bash
python tests/selftest.py
# 也可对已运行实例执行：
RAIN_BASE_A=http://127.0.0.1:8081 RAIN_BASE_B=http://127.0.0.1:8082 python tests/selftest.py
```

覆盖：参数校验、幂等/异文拒收、修订号顺序、成员齐全门禁、概率与分母数学、全缺测空值、
更正即过期与历史可查、双实例交错发布一致性、并行发布幂等、故障注入下无半份产品、
慢计算不阻塞其他时次入仓、Swagger/OpenAPI 可用性；以及实测核验专项——
首接纳锁定现行产品/无产品拒收、实测修订幂等与异文拒收、Brier 均方误差与有效格过滤、
零有效格空分数、成员更正不改锁定对象、实测更正旧核验立即失效且重算仍评估锁定产品、
双实例交错更正/核验一致性、并行核验幂等、核验故障注入无半份结果、慢核验不阻塞入仓。

## 5. 目录结构

```
app/
  main.py       FastAPI 路由、统一错误处理、启动建表
  schemas.py    Pydantic 请求模型与结构校验
  service.py    登记/提交/发布/实测/核验事务逻辑与并发控制
  storage.py    网格序列化、内容哈希、NumPy 概率与 Brier 计算
  db.py         SQLite 连接、WAL、schema（含 observations/verifications 表）
  errors.py     统一 API 异常
tests/selftest.py   端到端自测（双实例）
Dockerfile / docker-compose.yml / requirements.txt / run_local.sh
```
