# VLABench Pi0.5 评测排障记录（2026-03-11）

本文总结今天在 `VLABench + openpi` 跑 `pi0.5` 评测时遇到的问题、根因、修复方式，以及为什么这么修。  
适用于当前机器路径：

- `/data1/haodong2/weilin/red_bird/VLABench`

---

## 1. 依赖安装失败：`pyav>=12.0.5` 找不到

### 现象

- `pip install -r requirements.txt` 报：
  - `No matching distribution found for pyav>=12.0.5`

### 根因

- 旧版 `lerobot` 某个 commit 把依赖写成了 `pyav`（错误/过时），而 PyPI 正确包名是 `av`。
- 项目里把 `lerobot` 锁到了旧 commit，导致每次重装都会复现。

### 解决

- 把 `lerobot` 源提交改到修复后的 commit：
  - `ed83cbd4f2091a3e97cdd0c48cc657020c037240`
- 同时更新 lock：
  - `uv lock --upgrade-package lerobot`
  - `uv sync`

### 为什么这样修

- 不是本机网络偶发，而是依赖源本身有问题；必须切到修复 commit 才能根治。

---

## 2. 改了上层 `requirements.txt` 但 `uv sync` 仍报错

### 现象

- 已修改 `VLABench/requirements.txt`，但 `openpi` 里 `uv sync` 仍在拉旧 `lerobot`。

### 根因

- `openpi` 用的是自己的依赖入口：
  - `openpi/pyproject.toml`
  - `openpi/uv.lock`
- 不读取上层 `requirements.txt`。

### 解决

- 修改 `openpi/pyproject.toml` 中 `[tool.uv.sources]` 的 `lerobot` 提交。
- 执行 `uv lock --upgrade-package lerobot` 刷新锁文件。

### 为什么这样修

- 要修“实际被执行的依赖链”，不是修“看起来相关但没被用到的文件”。

---

## 3. 多卡误启动 / 跑错 GPU / 显存 OOM

### 现象

- `serve_policy.sh` 一启动就起 4 个服务。
- 你想用 GPU 2，实际跑到了 GPU 0，随后 OOM。

### 根因

- 脚本按 `nvidia-smi` 自动检测卡数并循环起进程。
- 脚本内部有 `CUDA_VISIBLE_DEVICES=${gpu_id}`，会覆盖外部环境。
- 曾出现环境变量拼写错误：`CUDE_VISIBLE_DEVICES`（应为 `CUDA_VISIBLE_DEVICES`）。

### 解决

- 将 `serve_policy.sh` 改为单卡运行（固定 1 个进程）。
- 固定到 GPU 2。
- 加入：
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`

### 为什么这样修

- 防止 JAX 在错误 GPU 上抢占显存。
- 关闭预分配可显著降低初始化阶段显存峰值，减少 OOM。

---

## 4. 权限错误：写入 `/inspire/...` 失败

### 现象

- `PermissionError: [Errno 13] Permission denied: '/inspire'`

### 根因

- `openpi/vlabench_scripts/setup_env.sh` 写死了作者机器路径（`/inspire/...`）。

### 解决

- 改为本机可写缓存目录：
  - `OPENPI_DATA_HOME=~/.cache/openpi_data`
  - `HF_LEROBOT_HOME=~/.cache/huggingface/lerobot`
  - `HF_HOME=~/.cache/huggingface`

### 为什么这样修

- 下载与缓存路径必须可创建、可写，才能初始化 tokenizer / 资源缓存。

---

## 5. `multi_run_vlabench.sh` 系统性不适配

### 现象

- conda 激活报错、路径不对、端口/GPU 混乱、后处理脚本报错。

### 根因

- 脚本包含大量作者本地硬编码（conda 环境名、绝对路径、并发策略等）。

### 解决（已做）

- 去掉或规避不必要的 conda 激活依赖。
- 固定单卡单进程。
- 固定端口与 serving 对齐（`8009`）。
- 修正：
  - `VLABENCH_ROOT=/data1/haodong2/weilin/red_bird/VLABench/VLABench`
  - `PYTHONPATH=/data1/haodong2/weilin/red_bird/VLABench`
- 暂时注释与当前流程无关的收尾脚本（避免误报）。

### 为什么这样修

- 先保证评测主链路稳定，再考虑恢复并行和报表自动化。

---

## 6. `ModuleNotFoundError` 连续缺包（`open3d` / `colorlog` / `rrt_algorithms`）

### 现象

- 评测进程不断报缺包。

### 根因

1. `uv run` 使用 `openpi/.venv`，不会自动使用外层 conda 的包。  
2. `VLABench/setup.py` 的依赖声明不完整（仅核心依赖）。  
3. `rrt-algorithms` 发布包结构不完整，缺失 `rrt/` 子模块。

### 解决

- 在 `openpi/.venv` 补齐评测依赖（一次性安装常用缺失包）。
- 将 `rrt-algorithms` 改为源码 editable 安装（绕过残缺 wheel）。

### 为什么这样修

- “逐个报错逐个装”效率低；一次补齐 + 修复坏包来源更稳定。

---

## 7. MuJoCo 编译失败：`counter` 网格法向/拓扑异常

### 现象

- 报错：
  - `faces of mesh 'counter/counter' have inconsistent orientation`

### 根因

- `counter.obj` 资产本身网格存在问题（非流形/自交等）。

### 解决

- 对资产进行修复并保留备份：
  - 原文件：`.../counter.obj`
  - 备份：`.../counter.obj.bak`
- 之后 `load_env('add_condiment')` 可正常通过。

### 为什么这样修

- 这是资产质量问题，不是依赖或命令参数问题；修资产是最直接方案。

---

## 8. “看起来卡住在 0/50”

### 现象

- 终端反复出现 TensorFlow/JAX 初始化日志，进度条似乎停在 `0/50`。

### 根因

- 多数是日志噪音 + 首个 episode 慢。
- 某些情况下还会重复启动多个 `eval.py` 进程，互相抢资源。

### 解决

- 先确认是否真卡死：
  - 看 `eval.py` 进程是否在跑
  - 看 `videos/` 是否持续新增
  - 看 `metrics.json` 是否更新
- 避免重复启动；必要时先清理旧评测进程：
  - `pkill -f "examples/vlabench/eval.py"`

### 为什么这样修

- 先区分“慢”与“死锁”，避免误判并重复启动导致更慢。

---

## 当前推荐启动方式（最稳）

### 1) 先启动 policy server（单卡）

```bash
cd /data1/haodong2/weilin/red_bird/VLABench/openpi
bash vlabench_scripts/serve_policy.sh pi05_ft_vlabench_primitive /home/haodong2/weilin/red_bird/VLABench/checkpoint
```

### 2) 另一个终端启动评测（先单任务）

```bash
cd /data1/haodong2/weilin/red_bird/VLABench/openpi
bash vlabench_scripts/multi_run_vlabench.sh /home/haodong2/weilin/red_bird/VLABench/pi_result --track track_1_in_distribution --task add_condiment
```

---

## 结果文件在哪里看

- 汇总指标：
  - `/home/haodong2/weilin/red_bird/VLABench/pi_result/<track>/metrics.json`
- 单任务细节：
  - `/home/haodong2/weilin/red_bird/VLABench/pi_result/<track>/<task>/detail_info.json`
- 回放视频：
  - `/home/haodong2/weilin/red_bird/VLABench/pi_result/<track>/<task>/videos/*.mp4`

---

## 仍可能看到但可忽略的日志

- TensorFlow/JAX 的 cuDNN/cuBLAS “register twice” 警告
- CPU feature 提示（AVX/FMA）

这些通常不是致命错误；只要评测进度和结果文件在推进即可。
