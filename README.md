# Hermes Vector Engine ⚡

<div align="center">

**面向多智能体（Multi-Agent）与跨平台拓扑的工业级本地语义向量检索与智能知识中枢**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![FAISS FlatIP](https://img.shields.io/badge/FAISS-FlatIP%20(<150μs)-orange.svg)](https://github.com/facebookresearch/faiss)
[![BGE-M3](https://img.shields.io/badge/Embedding-BGE--M3%201024d-blueviolet.svg)](https://huggingface.co/BAAI/bge-m3)
[![Multi-Platform](https://img.shields.io/badge/Compute-CUDA%20%7C%20Metal%20%7C%20Vulkan%20%7C%20CPU-red.svg)](https://vulkan.org)

[English Documentation (README_EN.md)](README_EN.md) · [架构白皮书 (ARCHITECTURE.md)](ARCHITECTURE.md) · [BGE-M3 部署指南](references/bge-m3-vulkan-setup.md) · [混合检索调优](references/hybrid-search-tuning.md)

</div>

---

## 🌟 1. 项目愿景与设计哲学 (Vision & Philosophy)

在自主智能体（Autonomous AI Agent）与大语言模型系统中，**长短期记忆中枢与知识检索（Retrieval-Augmented Generation, RAG）** 是决定 Agent 能否长期进化、稳定解决复杂长链路任务的“海马体”。

**Hermes Vector Engine** 是一套专为 Multi-Agent 生态、复杂异构计算拓扑（如 Linux 服务器、macOS Metal、Windows 以及 WSL2 跨系统环境）打造的**高性能、高可靠、全离线本地知识检索中枢**。

### 核心设计原则
* 🔒 **100% 本地离线与零成本**：由本地 `BGE-M3`（1024 维高维稠密向量特征）驱动，**零外部 API 费用、零 Token 消耗、零私密数据外泄**。
* ⚡ **极速嵌入式双层架构**：`SQLite (WAL) + FAISS (FlatIP)` 原生内嵌，无需启动沉重的独立向量数据库守护进程，毫秒级冷启动。
* 🧩 **语法无损感知切块**：内置 Markdown 块级 AST 解析器，代码块与 Markdown 表格 100% 结构闭合，杜绝粗暴截断。
* 🌐 **多租户与多智能体物理隔离**：原生支持多配置档案（Multi-Profile）命名空间，支持跨 Profile 索引扫描与精准来源追溯。
* 🛡️ **高置信度防幻觉门限**：向量余弦检索内置硬过滤门限（`min_score >= 0.70`），自动剔除弱关联噪音；配合 Agent 层的**级联兜底检索模式（Tiered Fallback）**，兼顾精准度与 GPU 算力开销。

---

## 💥 2. 传统 Agent 向量检索的 3 大工业痛点

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 传统 RAG / 向量方案的局限                                │
├─────────────────────────┬──────────────────────────────┬───────────────────────────────┤
│ 💥 痛点 1: 遗忘与上下文断崖 │ 💥 痛点 2: 代码与表格截断破坏  │ 💥 痛点 3: 多智能体记忆串味污染 │
│ 传统系统对历史会话做固定天数│ 传统按字符切块（如 500 字）， │ 多个 Agent 角色共用同一个平面  │
│ 截断，导致关键历史决策彻底  │ 将完整的函数代码或 Markdown  │ 向量空间，导致不同任务的记忆   │
│ 丢失，模型频繁陷入复读幻觉。│ 表格腰斩，表头与语法完全丢失。│ 互相污染，无法精确溯源。      │
└─────────────────────────┴──────────────────────────────┴───────────────────────────────┘
```

---

## 🏛️ 3. 全景系统架构图谱 (System Architecture)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. 扫描与摄取层 (Multi-Profile Data Ingestion Layer)                                   │
│    ├── Primary Profile (~/.hermes/) ──► 活跃会话 / 归档历史 / 持久记忆 / 技能与深度手册   │
│    └── Named Profiles (~/.hermes/profiles/*) ──► 命名档案 / 隔离会话 / 施工计划库        │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (1. 遍历扫描 + AST 智能切块 + 命名空间打标)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. 索引总控与防御层 (Python Indexer: scripts/index_all.py)                              │
│    ├── 隐私安全动态哨兵 (Privacy Sentinel: 实时拦截敏感/无痕会话入库)                   │
│    └── 4 大 SQLite 增量追踪账本 (Trackers: 比对 mtime/size/count，无变更 0 秒跳过)      │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (2. HTTP 流式批处理请求 /embedding, Batch=32)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. 硬件推理加速层 (Heterogeneous Compute Acceleration)                                 │
│    ├── Windows / WSL2: Windows 原生 Vulkan GPU 卸载 (-ngl 99 全量加速, 端口 :8081)     │
│    ├── Linux: NVIDIA CUDA / ROCm 硬件直通加速                                           │
│    └── macOS / CPU: Apple Metal (MPS) 或 AVX-512 高效指令集                            │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (3. 毫秒级回传 1024-dim Float32 二进制 Blob)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. 双层存储与空间索引 (Dual-Storage Architecture in SQLite + FAISS)                     │
│    ├── SQLite 库: vector_store.db (持久化存储文本、4096 字节 Blob、元数据与溯源打标)    │
│    └── FAISS 索引: vector_index.faiss (FlatIP 内存级精确余弦相似度空间, 检索 < 150μs)   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (4. 稠密向量余弦检索 + --min-score 0.70 门限过滤)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. 消费与交互层 (Runtime Consumers & Integration)                                      │
│    ├── CLI 独立检索工具: index_all.py search (支持 --kind 分类与 --min-score 门限过滤) │
│    ├── Agent 级联混合检索: session_search (FTS5 精确匹配优先 + 向量置信度自适应兜底)   │
│    └── 独立 Agent 接入模块: examples/standalone_rag_demo.py                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 4. 7 大核心技术支柱深度解析 (Architectural Pillars)

### 支柱 1：Markdown & Code AST 语法感知切块 (AST-Aware Chunking)

传统分块工具按纯字符数生硬截断，导致代码块与表格在中间断裂。Hermes Vector Engine 内置块级状态机解析器：

```text
❌ 传统按字符切块 (Fixed-Size Chunking):
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│ def query_database(sql: str):        │  │     rows = cursor.fetchall()         │
│     conn = sqlite3.connect(db_path)  │  │     return rows                      │  <-- 语法腰斩，
│     cursor = conn.cursor()           │  │                                      │      丢失代码闭合
└──────────────────┬───────────────────┘  └──────────────────────────────────────┘
                   └─ Chunk 1 ─── 断裂 ───► Chunk 2 ─┘

✅ Hermes AST 语法感知切块 (AST-Aware Chunking):
┌────────────────────────────────────────────────────────────────────────┐
│ ### Database Access Layer                                              │  <-- 自动保留标题定语
│ ```python                                                              │
│ def query_database(sql: str):                                          │  <-- 代码块原子闭合
│     conn = sqlite3.connect(db_path)                                   │
│     cursor = conn.cursor()                                             │
│     rows = cursor.fetchall()                                           │
│     return rows                                                        │
│ ```                                                                    │
└────────────────────────────────────────────────────────────────────────┘
```

* **原子级保护**：代码围栏（```` ``` ````）与表格结构（`|...|`）被严格识别为最小原子块，绝不在块内切断；
* **层级上下文继承**：切块自动继承最近一级 Markdown 标题（如 `### Database Access Layer`），确保检索时携带语境。

---

### 支柱 2：多配置档案与多租户原生物理隔离 (Multi-Profile Isolation)

通过原生支持多 Profile，彻底杜绝多角色/多环境的数据串味：

| 资产类型 | `default` 主档案标签 | `ops` 等命名档案标签格式 | 检索分类 (`--kind`) |
| :--- | :--- | :--- | :---: |
| **活跃会话** | `active_session:{sid}:b{bi}_c{ci}` | `profile:ops:active_session:{sid}:b{bi}_c{ci}` | `session` |
| **归档会话** | `session:{name}:b{bi}_c{ci}` | `profile:ops:session:{name}:b{bi}_c{ci}` | `session` |
| **技能总纲** | `skill:{name}:body` | `profile:ops:skill:{name}:body` | `skill` |
| **深度手册** | `skill:{name}:ref:{ref_name}` | `profile:ops:skill:{name}:ref:{ref_name}` | `skill` |
| **持久记忆** | `memory:{file}:entry_{idx}` | `profile:ops:memory:{file}:entry_{idx}` | `memory` |
| **施工计划** | `plan:{filename}` | `profile:ops:plan:{filename}` | `plan` |

---

### 支柱 3：5 维异构知识资产全域覆盖体系

系统将智能体运行时的所有形态知识资产归纳为 5 维异构模型，统一流式索引：

1. **会话全量历史 (`sessions`)**：直连会话存储引擎（如 `state.db`），突破传统天数截断断崖，支持活跃对话与归档 JSONL。
2. **多级技能与深度参考手册 (`skills`)**：递归解析技能总纲（`SKILL.md`）以及多级子目录中的技术手册与排障指南（`references/*.md`）。
3. **长期持久记忆系统 (`memory`)**：包含 Agent 系统常识、操作契约与用户画像定义。
4. **工程架构与施工计划库 (`plans`)**：工程架构计划图谱、模块契约与阶段性任务依赖。
5. **动态隐私安全防御哨兵 (`privacy sentinel`)**：可插拔的安全熔断机制（如无痕模式标记或环境变量拦截），在向量化前置阶段自动跳过敏感会话，防止私密数据落库。

---

### 支柱 4：SQLite 二进制 Blob + FAISS FlatIP 双层存储

* **SQLite 主表 (`vectors`)**：存储原始切块文本、MD5 校验和、来源标记以及 **4096 字节的 Float32 二进制 Blob** (`1024 * 4 bytes`)。
* **FAISS 内存索引 (`vector_index.faiss`)**：加载 Float32 向量并执行 `IndexFlatIP` 精确点积（L2 归一化后等价于余弦相似度），在大规模向量空间内检索仅耗时 **< 150 微秒**。
* **零额外服务**：无需部署臃肿的外部独立向量数据库，进程内直读直写，备份仅需拷贝 2 个文件。

---

### 支柱 5：级联自适应混合检索模式 (Tiered Fallback Hybrid Search)

在实际 Agent 集成中，系统推荐采用 **“级联自适应兜底架构（Tiered Fallback Pattern）”**，兼顾精准度与 GPU 算力效率：

```
                    [Agent 收到用户查询 / 排障请求]
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │ 步骤 1: FTS5 关键词精确快速检索   │ ──► 命中足够结果？ ──► [直接返回 (0 GPU 开销)]
                └─────────────────┬─────────────────┘         (Yes)
                                  │ (No: 结果不足 / 概念模糊)
                                  ▼
                ┌───────────────────────────────────┐
                │ 步骤 2: 触发向量引擎置信兜底补充  │
                │ • index_all.py search --kind ... │
                │ • 强制过滤 min_score < 0.70 噪音  │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │ 步骤 3: 语义高置信结果补充注入上下文│
                └───────────────────────────────────┘
```

* **精准词优先**：精确错误码、函数名、配置项优先走 FTS5 毫秒级命中，避免向量漂移；
* **语义置信兜底**：当用户提问概念模糊（如“核显”对应“780M / GPU”）或关键词稀疏时，自动激活向量引擎补充高置信语义切块。

---

### 支柱 6：置信度硬门限过滤机制 (`min_score = 0.70`)

为彻底消除 Agent 检索中的“虚假相关幻觉”，系统设置了标准置信度阶梯：

```text
得分区间               契合等级              系统策略与动作
┌────────────────────────────────────────────────────────────────────────┐
│ 0.85 ~ 1.00  │ 🌟 极强相关 / 精准命中 │ 直接作为强置信证据注入上下文           │
│ 0.75 ~ 0.85  │ 🔥 高度相关           │ 作为关键技术上下文参考                 │
│ 0.70 ~ 0.75  │ 💡 中度相关           │ 作为辅助背景提示                       │
├────────────────────────────────────────────────────────────────────────┤
│ < 0.70       │ ❄️ 低置信噪音 (截断)   │ 🛡️ 自动拦截过滤，严禁进入大模型上下文  │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 支柱 7：流式分批事务与存储生命周期治理 (WAL + VACUUM)

* **流式分批提交**：以 30 个文件 / 32 个 Chunk 为事务分批写入 SQLite，避免长时间占用写锁，断点自愈续跑。
* **空间回收治理**：内置 `vacuum` 指令与周常 WAL Checkpoint 垃圾回收机制，有效清理物理磁盘碎片。

---

## 🖥️ 5. 跨平台与异构算力适配指南 (Compute Topologies)

Hermes Vector Engine 兼容多种异构算力后端，通过环境变量 `HERMES_EMBED_URL` 即可对接任意本地或远程端点：

```
┌────────────────────────────────────────────────────────────────────────┐
│                         跨平台算力拓扑支持矩阵                          │
├──────────────────┬──────────────────┬──────────────────────────────────┤
│ 部署操作系统     │ 推荐计算后端     │ 启动方式与核心优势               │
├──────────────────┼──────────────────┼──────────────────────────────────┤
│ Linux / Server   │ NVIDIA CUDA      │ llama-server --ngl 99 (高并发)   │
│ macOS (Apple Sil)│ Apple Metal/MPS  │ llama-server --ngl 99 (统一内存) │
│ Windows / WSL2   │ Windows Vulkan   │ Windows Vulkan 宿主机算力桥接    │
│ 通用轻量环境     │ 纯 CPU (AVX2/512)│ llama-server -ngl 0 (零硬件门槛) │
└──────────────────┴──────────────────┴──────────────────────────────────┘
```

> **WSL2 + Windows 跨系统拓扑说明**：
> 在 Windows 11 + WSL2 架构下，AMD Radeon 等显卡在 WSL2 内部缺乏 ROCm 驱动支持。通过在 Windows 宿主机运行 Vulkan 后端 `llama-server.exe :8081`，并在 WSL2 中配置镜像网络（`networkingMode=mirrored`），实现 **0.2ms 超低延迟的跨系统 GPU 硬件全量加速**。

---

## 🚀 6. 开发者快速上手与集成模式 (Quick Start)

### 模式 A：5 分钟独立体验 Demo (Standalone)

无需安装完整 Agent 环境，直接运行独立验证脚本：

```bash
# 1. 克隆仓库与安装依赖
git clone https://github.com/GenmetsuWenxuePress/hermes-vector-engine.git
cd hermes-vector-engine
pip install -r requirements.txt

# 2. 运行独立 RAG 架构 Demo
python3 examples/standalone_rag_demo.py
```

---

### 模式 B：接入本地 GPU 向量服务并全量索引

#### 第 1 步：启动本地 BGE-M3 嵌入服务
下载预编译的 [llama.cpp](https://github.com/ggerganov/llama.cpp/releases) 与 `bge-m3-Q4_K_M.gguf` 权重：
```bash
llama-server \
  -m models/bge-m3-Q4_K_M.gguf \
  --embeddings \
  --host 127.0.0.1 \
  --port 8081 \
  -b 2048 -ub 2048 -ngl 99 -t 4
```

#### 第 2 步：执行全域增量同步
```bash
# 执行增量索引（未变动文件 0 秒秒跳过）
python3 scripts/index_all.py

# 查看向量库统计与各 Profile 占比分布
python3 scripts/index_all.py stats
```

#### 第 3 步：执行精准语义检索与过滤
```bash
# 1. 全局语义检索（默认自动过滤 < 0.70 噪音）
python3 scripts/index_all.py search "7步升级流水线" 5

# 2. 仅在技能参考手册中精准检索，设置 0.75 强门限
python3 scripts/index_all.py search --kind skill --min-score 0.75 "WARP 链式代理" 3

# 3. 检索施工图计划库
python3 scripts/index_all.py search --kind plan "self improvement" 2

# 4. 数据库碎片压缩与空间整理
python3 scripts/index_all.py vacuum
```

---

## ⚖️ 7. 架构设计权衡与深度 FAQ (Design Trade-offs)

### Q1: 为什么选择 FAISS FlatIP 而不是 HNSW 索引？
* **量级决定架构**：在个人知识库与 Agent 记忆场景（< 10 万条向量）下，`FlatIP`（暴力精确内积计算）在 3.5 万条向量空间内的检索延迟仅为 **< 0.15 毫秒**。
* **100% 绝对精确召回**：HNSW 属于近似最近邻（ANN），存在精度损失且需要显著的建图时间与内存开销；`FlatIP` 提供了零召回损失与极低的内存占用。

### Q2: 为什么分块重叠设为 0 (`OVERLAP_CHARS = 0`)？
* **语义纯度至上**：实测表明，在拥有 Markdown AST 语法感知分块的前提下，每个块已具备完备的语义边界与标题上下文。生硬设置 Overlap 反而会产生重复冗余的切块碎片，稀释 Top-1 命中的置信度。

### Q3: 为什么坚决使用 SQLite 二进制 Blob 而非外部独立向量库（Milvus/Qdrant）？
* **嵌入式轻量原则**：AI Agent 追求高内聚与极低运维负担。SQLite + FAISS 原生单文件持久化，零网络 IO 开销，随启随用，整库迁移只需打包单个文件。

---

## 📊 8. 生产环境规模验证与物理基准 (Production Scale Benchmarks)

以下数据基于真实生产环境（涵盖 850+ 篇深度技术手册、全量历史会话与多 Profile 资产，共计 35,189 条 1024 维向量空间）实测验证：

| 评估指标 | 生产环境实测数据 (AMD Ryzen 7 8845H / Radeon 780M / 32GB) |
| :--- | :--- |
| **FAISS FlatIP 检索延迟** | **< 150 微秒 (0.15ms)**（在 35,189 条 1024 维向量空间内） |
| **GPU 特征提取吞吐** | **~35 Chunks / 秒**（Vulkan 硬件加速） |
| **3.5 万条增量比对耗时** | **< 0.05 秒**（SQLite Tracker 哈希比对） |
| **单次 VACUUM 碎片回收** | 释放 **8.40MB** 磁盘空间 |
| **宿主机内存常驻占用** | **~1.36 GB**（BGE-M3 模型加载后稳定常驻） |

---

## 🧭 常用指令速查 (Cheat Sheet)

```bash
# 检查嵌入服务健康度
curl -s --noproxy '*' http://127.0.0.1:8081/health

# 运行自动化测试套件
python3 tests/test_vector_engine.py

# 运行独立架构演示 Demo
python3 examples/standalone_rag_demo.py
```

---

## 📄 开源协议 (License)

本项目遵循 [MIT License](LICENSE) 开源协议。  
欢迎自由 Fork、Star 与提交 Pull Request！
