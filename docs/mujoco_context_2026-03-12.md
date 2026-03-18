# VLABench MuJoCo 上下文整理

日期：2026-03-12

本文只整理这次 `VLABench + openpi + pi0.5` 评测过程中，和 **MuJoCo / dm_control / 机器人逆运动学 / 资产 XML / 物理仿真稳定性** 有关的上下文，方便后续继续排查。

---

## 1. 当前评测链路里 MuJoCo 在哪里起作用

当前评测不是模型直接输出机器人关节角，而是：

1. 模型输出末端执行器目标：
   - `pos`
   - `euler`
2. 评测器把末端目标交给机器人 IK 求解：
   - `env.robot.get_qpos_from_ee_pos(...)`
3. IK 返回关节角后，再交给 MuJoCo / dm_control 做物理仿真 step。

也就是说，当前评测链路里和 MuJoCo 相关的问题主要分成三层：

1. **资产 / XML 构建问题**
   - XML 不合法
   - mesh 几何坏
   - MuJoCo 编译失败
2. **IK 问题**
   - 目标位姿不可达
   - 99 步没收敛
3. **物理仿真稳定性问题**
   - step 过程中数值爆炸
   - `PhysicsError`

---

## 2. 这次见到的 MuJoCo 相关日志，分别代表什么

### 2.1 `Failed to converge after 99 steps: err_norm=...`

这是 **IK 警告**，不是 MuJoCo XML 编译错误。

含义：
- 模型给了一个末端位姿目标
- IK 尝试 99 步后没能找到足够精确的关节角解
- 但系统仍然可能继续执行“近似解”

这通常说明：
- 目标姿态不太可达
- 机械臂接近工作空间边界
- 手腕姿态不舒服
- 动作输出对当前场景几何不够友好

注意：
- 它是 **warning，不一定导致当前 episode 立刻失败**
- 但 warning 多时，更容易出现抓取失败、控制偏差、甚至后续仿真不稳定

经验解释：
- `err_norm` 小（如 `0.001 ~ 0.01`）：一般是“差一点没收敛”
- `err_norm` 中等（如 `0.03 ~ 0.15`）：可能影响动作精度
- `err_norm` 大（如 `0.3 ~ 0.5+`）：很可能明显不可达

---

### 2.2 `Compile error raised by Mujoco`

这是 **环境构建阶段** 的致命错误，说明 MuJoCo 在读取 XML / mesh 时就失败了。

常见原因：
- XML 非法
- 属性值为空或格式错误
- mesh 面法向/拓扑异常

它和 IK warning 不一样：
- IK warning 发生在动作执行阶段
- MuJoCo compile error 发生在环境初始化阶段

---

### 2.3 `Physics state is invalid` / `mjWARN_BADQACC`

这是 **仿真过程中数值不稳定** 的致命错误。

典型日志：
- `Nan, Inf or huge value in QACC`
- `Physics state is invalid`
- `PhysicsError`

含义：
- 当前物理状态已经发散
- 仿真步进不能继续
- 通常由激进动作、奇怪碰撞、异常姿态或接触导致

这个问题往往不是资产 XML 语法错误，而是：
- 模型输出动作过激
- IK 不稳定后继续执行
- 接触状态异常
- 某些 episode 场景本身更容易诱发不稳定

---

## 3. 这次已经确认并修复的 MuJoCo 资产问题

### 3.1 `counter.obj` 网格坏

现象：
- MuJoCo 报：
  - `faces of mesh 'counter/counter' have inconsistent orientation`

原因：
- `counter.obj` 网格本身存在非流形 / 自交 / 朝向异常

处理：
- 已对该网格做清理与重写
- 保留了备份文件：
  - `counter.obj.bak`

影响：
- 这一类错误会导致某些 episode 在构建环境时直接失败

---

### 3.2 `placemat_5.xml` 非法空属性

现象：
- MuJoCo 报：
  - `XML Error: problem reading attribute 'euler'`

原因：
- 文件里存在非法属性：
  - `euler=""`

处理：
- 已改为：
  - `euler="0 0 0"`

影响：
- 修复后，命中 `placemat_5` 的 episode 不会再因为这个 XML 直接失败

---

### 3.3 额外发现的 4 个坏 XML

在全量扫描 `VLABench/VLABench/assets` 的 1019 个 XML 后，发现并修复了 4 个 parse error 文件：

1. `assets/obj/meshes/snacks_and_drinks/wine_bottle/wine_bottle.xml`
2. `assets/obj/meshes/snacks_and_drinks/wine_bottle/wine_bottle_red_main.xml`
3. `assets/obj/meshes/tablewares/teapots/moka_pot/moka_pot.xml`
4. `assets/obj/meshes/tablewares/teapots/moka_pot/moka_pot_handle.xml`

问题本质：
- `worldbody/body` 标签嵌套不合法
- 存在标签不匹配

修复后结果：
- 当前资产目录 XML 全量校验通过：
  - `xml_count = 1019`
  - `error_count = 0`

注意：
- 这保证了 XML 结构合法
- 但不保证每个 mesh 几何都绝对没有问题

---

## 4. 现在如何判断一条 MuJoCo 日志到底严不严重

### 可暂时忽略 / 记录但不必立刻停

1. `Failed to converge after 99 steps`
2. `glfw ... DISPLAY environment variable is missing`
3. TensorFlow / cuDNN / cuBLAS 重复注册日志

这些通常不是让整个任务停掉的直接原因。

---

### 需要重点关注

1. `Compile error raised by Mujoco`
2. `XML Error: ...`
3. `Physics state is invalid`
4. `mjWARN_BADQACC`
5. Python `Traceback` 最终抛出 `ValueError` / `PhysicsError`

这些都是真正会让 episode 失败、甚至让环境无法继续运行的问题。

---

## 5. 为什么权重针对这个数据集微调过，仍然会有 IK warning

这并不矛盾。

原因：

1. 模型学的是动作分布，不是严格的运动学可达解空间
2. 数据中的“可执行动作”经过了控制器/场景约束，而模型输出是近似模仿
3. 小物体、平面物体、姿态要求高的任务更容易触发 IK 边界问题
4. IK warning 不一定导致当前 episode 失败，只是说明目标位姿不理想

所以：
- 某些 task/track 正确率不错
- 仍然可能伴随很多 IK warning

这是机器人策略评测里比较常见的现象。

---

## 6. 当前评测器对 MuJoCo / PhysicsError 的处理方式

当前评测器已经改过：

### 以前
- exception episode 会被直接跳过
- 可能导致结果统计偏乐观

### 现在
- exception episode 也会写进 `detail_info.json`
- 并且按失败计入指标
- 同时带异常标签，便于后续排查

现在每个 episode 都会带这些字段：

- `exception`
- `exception_type`
- `exception_message`
- `traceback`

这样就能区分：

1. 正常失败
2. IK warning 但继续执行
3. MuJoCo / PhysicsError 真异常

---

## 7. 目前最值得注意的潜在问题

当前最值得继续观察的是：

### 7.1 IK `success` 结果未被显式用于动作过滤

虽然机器人接口会返回：
- `success`
- `qpos`

但评测器当前仍然主要使用 `qpos` 去执行动作。  
也就是说，某些 IK 未收敛动作仍会继续送进仿真，这可能增加：

- 末端偏差
- 不稳定接触
- PhysicsError 概率

这不是环境配置错误，而是当前评测逻辑本身的设计取舍。

---

### 7.2 某些 episode 仍可能因为仿真数值不稳定失败

即使资产 XML 已经清理过，后续仍可能出现：
- `mjWARN_BADQACC`
- `Physics state is invalid`

这类问题更偏向：
- 控制输出
- 任务几何
- 接触关系

而不是静态资产文件格式问题。

---

## 8. 当前建议的排查顺序

后续如果再出 MuJoCo 相关问题，建议按这个顺序判断：

1. **先看是不是 XML / asset compile error**
   - 如果是，优先修资产
2. **再看是不是 IK warning**
   - 如果只是 warning，先统计是否影响成功率
3. **最后看是不是 PhysicsError**
   - 这通常说明动作/接触/仿真已经发散

---

## 9. 当前阶段的结论

截至 2026-03-12，MuJoCo 相关上下文可以总结为：

1. 资产 XML 结构层面的问题已经做过一轮较完整清理
2. `counter` 网格和 `placemat_5` XML 这类明确坏资产已修
3. 现在更多出现的是：
   - IK 不收敛 warning
   - 个别 episode 的仿真不稳定
4. 这说明当前瓶颈已经从“环境能不能跑起来”转向了：
   - 模型动作可达性
   - 仿真稳定性
   - 个别任务/episode 的几何与控制问题

---

## 10. 后续可做但尚未做的增强

如果后面继续优化 MuJoCo 相关稳定性，可优先考虑：

1. 在 `detail_info.json` 里进一步统计每个 episode 的 IK warning 次数
2. 当 IK `success=False` 时额外打 `ik_failed` 标签
3. 对模型输出做 workspace / orientation clamp
4. 对易炸任务做单独 smoke test
5. 对高频使用资产做 MuJoCo compile 级别批量冒烟测试

