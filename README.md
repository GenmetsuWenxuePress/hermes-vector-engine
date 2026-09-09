# Hermes Vector Engine ⚡

<div align="center">

**面向多智能体（Multi-Agent）与跨系统拓扑的工业级本地语义向量检索与智能知识中枢**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![FAISS FlatIP](https://img.shields.io/badge/FAISS-FlatIP%20(<150μs)-orange.svg)](https://github.com/facebookresearch/faiss)
[![BGE-M3](https://img.shields.io/badge/Embedding-BGE--M3%201024d-blueviolet.svg)](https://huggingface.co/BAAI/bge-m3)
[![Vulkan GPU](https://img.shields.io/badge/Hardware-Vulkan%20GPU%20Offload-red.svg)](https://vulkan.org)

[English Documentation](README_EN.md) · [架构白皮书](ARCHITECTURE.md) · [BGE-M3 部署指南](references/bge-m3-vulkan-setup.md) · [混合检索调优](references/hybrid-search-tuning.md)

</div>

---

## 🌟 项目概述与核心亮点

**Hermes Vector Engine** 是一套专为自主 AI Agent（如 Hermes、AutoGPT、本地大模型工作流）设计的高性能、高可靠、全离线本地知识检索中枢。

针对在 **WSL2 + Windows 宿主机跨系统架构** 下传统向量检索面临的*“依赖昂贵外部云端 API”、“AMD 显卡在 WSL2 中缺乏驱动支持”、“长文本分块粗暴截断代码/表格”、“多配置档案（Profile）相互污染”*等真实工业界痛点，本项目提出并实现了完整的端到端闭环方案：

* 🔒 **100% 本地离线与零成本**：由本地 `BGE-M3` (1024 维高维特征) 密集向量模型驱动，**零外部 API 费用、零 Token 消耗、零私密数据外泄**。
* ⚡ **GPU Vulkan 硬件级全量加速**：Windows 宿主机 Vulkan 计算后端实现 **99 层全量卸载 (`-ngl 99`)**，AMD Radeon 780M / 880M / 独立显卡均可获得 **3.5x+** 的吞吐提速，毫秒级响应。
* 🌐 **多配置档案原生隔离 (Multi-Profile Native)**：自动遍历发现 `default` 与所有命名档案（如 `ops`），通过复合命名空间与来源标签（`profile:<name>:...`）实现严格的物理隔离与针对性检索。
* 🧩 **语法感知智能切块 (AST-Aware Chunking)**：内置 Markdown 块级解析器，**原子级保护 ` ``` ` 代码块与 `| 表格 |` 结构**，彻底杜绝语法切断与表头丢失，每个片段自带上下文章节定语。
* 📚 **5 维全域知识深度覆盖**：全量覆盖**活跃会话记录（无 30 天截断断崖）、历史 JSONL 归档、长期持久记忆、850+ 技能深度参考手册 (`references/*.md`) 与工程施工图计划库 (`plans/*.md`)**。
* 🛡️ **流式分批事务与可自愈机制 (Streaming Batch Commits)**：以 30 文件 / 32 Chunk 为批次进行流式写入与即时存盘，进程中断不丢数据，支持断点自愈续跑。
* 🧹 **存储生命周期自动 GC**：内置 SQLite `VACUUM` 与 WAL Checkpoint 垃圾回收管理，定期整理碎片释放空间，数据库体积保持紧凑。

---

## 🏛️ 全景架构拓扑与数据流图谱

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. 扫描与摄取层 (Multi-Profile Data Ingestion Layer in WSL2)                           │
│    ├── default 档案 (~/.hermes/) ──► state.db / sessions/*.jsonl / memories / skills   │
│    └── ops 档案 (~/.hermes/profiles/ops/) ──► state.db / memories / skills / plans     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (1. 遍历扫描 + AST 智能切块 + 来源打标)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. 索引总控与防御层 (Python Indexer: scripts/index_all.py)                              │
│    ├── 无痕模式安全哨兵 (/tmp/.hermes-incognito-active: 实时拦截敏感会话入库)           │
│    └── 4 大 SQLite 增量追踪账本 (Trackers: 比对 mtime/size/count，无变更 0 秒跳过)      │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (2. HTTP 流式批处理请求 /embedding, Batch=32)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. 硬件推理加速层 (Windows 宿主机: llama-server.exe on Port :8081)                      │
│    ├── 计算后端: ggml-vulkan.dll (AMD Radeon 780M / 独显 GPU, -ngl 99 全量卸载)         │
│    └── 嵌入模型: BGE-M3 Q4_K_M (1024 维特征向量，显存/内存常驻仅 ~1.36GB)               │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (3. 毫秒级回传 1024-dim Float32 二进制 Blob)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. 双层存储与检索空间 (Dual-Storage Architecture in SQLite + FAISS)                     │
│    ├── SQLite 库: vector_store.db (持久化存储文本、4KB 二进制 Blob 与元数据)            │
│    └── FAISS 索引: vector_index.faiss (FlatIP 内存级精确余弦相似度计算, <150μs)         │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (4. 双轨混合召回: FTS5 关键词优先 + 向量置信补充)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. 消费与交互层 (Runtime Consumers)                                                    │
│    ├── 智能会话搜索工具: session_search (跨档案自动语义召回)                           │
│    └── CLI 运维检索工具: index_all.py search (支持 --kind 分类与 --min-score 门限过滤) │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 4 大维度深度演进对比

| 评估维度 | 改造前（传统基础版本） | 改造后（Hermes Vector Engine v4.0.0） | 带来的实质收益 |
| :--- | :--- | :--- | :--- |
| **维度 1：运行时集成** | • 多 Profile 来源解析截断为 `"profile"`<br>• 无置信度门限，低分噪音全量涌入 | • 正则精确提取 Session ID 与 Profile 命名空间<br>• 强制设置 **`min_score = 0.70`** 门限过滤 | • 跨档案会话无损召回<br>• 自动阻断低相关幻觉信息 |
| **维度 2：分块语义感知** | • 纯文本按字符截断<br>• 代码块与表格经常从中间被腰斩 | • **Markdown & Code AST 智能感知切块**<br>• 保持代码与表格原子完整性，继承标题定语 | • 代码语法与表格表头 100% 完整<br>• 技术文档检索准确率大幅提升 |
| **维度 3：多源知识覆盖** | • 仅覆盖基础 `SKILL.md`<br>• 活跃会话存在 30 天截断断崖 | • 全量覆盖 **850+ 技能深度参考手册 (`references/*.md`)**<br>• 纳入 **工程施工图计划库 (`plans/*.md`)** | • 遇到具体报错或历史计划时均可秒级命中<br>• 知识覆盖率提升 300%+ |
| **维度 4：存储与自愈** | • 单一大事务提交，大批量索引易超时<br>• 删除条目后数据库碎片不回收 | • **30 文件分批流式事务**，支持断点自愈续跑<br>• 内置 **`vacuum` 碎片整理**，与周常清理联动 | • 永不因超时丢失 GPU 算力<br>• 数据库长期保持紧凑高效 |

---

## 📦 5 维知识资产与来源打标规范

系统将所有知识按来源进行规范化前缀打标，严格保证多租户/多档案的溯源隔离：

| 资产大类 | 包含内容 | `default` 档案标签格式 | `ops` 及其他命名档案标签格式 | 检索 Kind |
| :--- | :--- | :--- | :--- | :---: |
| **活跃会话** | `state.db` 中的实时对话消息 | `active_session:{sid}:b{bi}_c{ci}` | `profile:{pname}:active_session:{sid}:b{bi}_c{ci}` | `session` |
| **归档会话** | `sessions/*.jsonl` 历史转储文件 | `session:{name}:b{bi}_c{ci}` | `profile:{pname}:session:{name}:b{bi}_c{ci}` | `session` |
| **技能手册** | 技能总纲 (`SKILL.md`) | `skill:{category}/{name}:{body/desc}` | `profile:{pname}:skill:{category}/{name}:{body/desc}` | `skill` |
| **深度参考** | 排障手册 (`references/*.md`) | `skill:{name}:ref:{ref_name}` | `profile:{pname}:skill:{name}:ref:{ref_name}` | `skill` |
| **系统记忆** | `MEMORY.md` 与 `USER.md` | `memory:{file}:entry_{idx}` | `profile:{pname}:memory:{file}:entry_{idx}` | `memory` |
| **工程计划** | 施工图架构计划 (`plans/*.md`) | `plan:{filename}` | `profile:{pname}:plan:{filename}` | `plan` |

---

## ⚙️ 双层存储与数据结构设计

### 1. SQLite 向量主表 (`vectors`) 结构
```sql
CREATE TABLE vectors (
    id TEXT PRIMARY KEY,        -- 12位 MD5 去重哈希指纹
    text TEXT NOT NULL,         -- 原始切块文本内容
    embedding BLOB NOT NULL,    -- 1024 维 Float32 二进制字节流 (4096 字节)
    source TEXT NOT NULL,       -- Profile 打标来源路径
    created_at REAL NOT NULL,   -- 索引创建时间戳
    kind TEXT DEFAULT 'session' -- 资产类别: session | memory | skill | plan
);
```

### 2. 增量跟踪账本 (Incremental Trackers)
SQLite 内部维护了 4 张极速比对表，记录每个文件的修改时间 (`mtime`)、字节大小 (`size`) 与会话消息数 (`message_count`)。扫描 3.5 万+ 条向量时，**增量比对在 0.05 秒内完成，未变动内容 0 读写、0 计算消耗**。

---

## 🚀 快速上手 (Quick Start)

### 第 1 步：启动 Windows 宿主机 GPU 向量引擎
下载 `llama-server` 与 `bge-m3-Q4_K_M.gguf` 权重文件，在 Windows 终端中运行：
```powershell
llama-server.exe `
  -m models/bge-m3-Q4_K_M.gguf `
  --embeddings `
  --host 127.0.0.1 `
  --port 8081 `
  -b 2048 -ub 2048 -ngl 99 -t 4
```

### 第 2 步：在 Linux / WSL 端运行增量索引
```bash
# 1. 触发全量多档案增量同步
python3 scripts/index_all.py

# 2. 查看当前向量库统计分布
python3 scripts/index_all.py stats
```

### 第 3 步：执行高置信度语义检索
```bash
# 1. 检索所有类别（默认过滤 < 0.70 噪音）
python3 scripts/index_all.py search "7步升级流水线" 5

# 2. 深度检索技能参考手册（指定 --kind skill 与置信度门限）
python3 scripts/index_all.py search --kind skill --min-score 0.75 "Vulkan 显卡卸载" 3

# 3. 检索工程施工图计划库
python3 scripts/index_all.py search --kind plan "self improvement" 2

# 4. 碎片整理与空间回收
python3 scripts/index_all.py vacuum
```

---

## 🧪 自动化测试套件

运行自带的 4 大维度全自动化回归单测套件：
```bash
python3 tests/test_vector_engine.py
```

实测输出：
```text
Testing Dimension 2: AST-Aware Chunking...
  ✓ 表格完整性验证: PASS (完整包含表头与所有数据行)
  ✓ 代码块原子性验证: PASS (完整包含声明、函数体与闭合标记)
Testing Dimension 1: Source Tagging & Regex Parsing...
  ✓ Source tagging and parsing test passed.
Testing Dimension 4: SQLite Integrity & Vacuum...
  ✓ SQLite Integrity & Vacuum test passed.

🎉 All tests passed successfully!
```

---

## 📊 性能基准实测数据 (Benchmarks)

| 性能指标 | 实测物理数据 (AMD Ryzen 7 8845H / Radeon 780M) |
| :--- | :--- |
| **FAISS FlatIP 检索延迟** | **< 150 微秒 (0.15ms)**（在 35,000+ 条向量空间内） |
| **GPU 批量特征提取吞吐** | **~35 Chunks / 秒**（Vulkan 硬件加速） |
| **增量状态比对耗时** | **< 0.05 秒**（扫描 300+ 技能与会话文件） |
| **数据库碎片回收率** | 单次 VACUUM 释放 **8.40MB** 碎片空间 |
| **宿主机内存常驻占用** | **~1.36 GB**（模型加载后保持稳定） |

---

## 🧭 常用运维指令速查表 (Cheat Sheet)

| 运维场景 | 执行指令 | 说明 |
| :--- | :--- | :--- |
| **检查 GPU 服务健康度** | `curl -s --noproxy '*' http://127.0.0.1:8081/health` | 返回 `{"status":"ok"}` 即正常 |
| **查看向量统计与占比** | `python3 scripts/index_all.py stats` | 输出各类型与 Profile 占比分布 |
| **手动触发全域增量同步** | `python3 scripts/index_all.py` | 自动扫描多 Profile 变动并编码入库 |
| **数据库碎片压缩整理** | `python3 scripts/index_all.py vacuum` | 释放 SQLite 已删除向量残留空间 |
| **按门限精准查技能手册** | `python3 scripts/index_all.py search --kind skill --min-score 0.80 "关键词"` | 过滤低置信噪音，输出最高相关度手册 |

---

## 📄 开源协议 (License)

本项目采用 [MIT License](LICENSE) 开源协议。
版权所有 (c) 2026 GenmetsuWenxuePress (Rodion)。欢迎自由 Fork、Star 与提交改进！
