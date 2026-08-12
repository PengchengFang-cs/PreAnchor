# Object2Anchor 项目状态

> 更新：2026-08-12。本文件是项目的权威进度记录，每个里程碑后更新。

## 一句话定位

给定场景/物体 + 语义 prompt + 人物状态，**用数学方法求解稀疏控制帧**（root anchor 为主），
驱动**冻结的、scene-blind 的**动作生成器完成人-场景交互；测试时用 GN 场景投影做小幅修正。
核心科学问题：**场景信息需要多少、通过哪条通道进入？** 投稿方向：CVPR（三个审稿 agent 一致结论）。

## 关键决策记录

| 决策 | 内容 |
|---|---|
| 数据集 | TRUMANS（唯一，不混合），/scratch/pf2m24/data/trumans（zirobtc 镜像，7.5G） |
| decoder（当前） | 公开 Kimodo-SMPLX-RP-v1（替身/基线）；自研 flow（TRUMANS 直训、teacher-forced 无 rollout、scene-blind）为最终 decoder |
| 预测器路线 | 数学优先：⓪动词→谓词分类（LLM/规则）→ ①解析几何求解 → ②非参数 anchor 库 → ③仅在实测失败处上小模型 |
| 运动学分类 | S 支撑转移（root-only）/ R 触及（root+hand）/ T 通过（waypoint 对）；动词多对一坍缩进三类 |
| 静态物体假设 | 第一期成立；open/pick 的物体运动推后 |
| anchor 定义 | commitment 时刻物体局部系下 root (x,y,z,yaw)；由身体侧谓词定义，不按物体类别定义 |

## 已完成

**M1 标签提取（完成）**：876 个 GT anchor（sit 778 / lie 98），绑定成功率 99.2%；
198 组多模态聚类，均值 2.17 模式/物体（长沙发最多 6 坐点）。产出 `out/anchors.jsonl`。

**M3 闭环 + 实验 E1（完成，2026-08-11）**：GT anchor → 约束 → Kimodo 生成 → 场景合规验证。
三臂结果（anchor #0，static_chair_03，座高 0.668m 偏高椅）：

| 臂 | 约束 | (x,z) | 坐高 | 穿透帧 |
|---|---|---|---|---|
| 1 root-(x,z,yaw) | 地面锚点+朝向 | 2.9cm | 默认 0.626（-4cm 穿座） | 58/240 |
| 1b +root_y 通道 | 24帧+CFG5.0 | 1.4cm | **被无视**（0.619 vs 目标 0.838） | 67/240 |
| 2 fullbody 终端关键帧 | 自举抬高 0.219m | 1.4cm | **0.838 误差 0.000** | **0/240** |

**E2 批量三臂（完成，2026-08-12）**：25 anchor（每物体组内中位代表，座高分位覆盖 0.264–0.912m，E1 的 s0 强制保留）× 三臂，75 次生成零失败（4.7s/次，rose09 GPU1）。协议统一 seed 1、CFG(2.0,5.0)、100 步、T=240；指标锚定目标点局部 0.5m 座面（多坐点家具只判目标模式），静态物体路径已补（sit anchor 中 536/778 是静态物体）。详表与读数 `out/e2/E2_summary.md`，逐条数据 `out/e2/e2_results.jsonl`。

| 臂 | y误差(带符号中位) | \|y\|<5cm | 穿透帧(均值) | 零穿透率 | xz<10cm |
|---|---|---|---|---|---|
| 1 root-(x,z,yaw) | -0.074 | 36% | 24.5 | 44% | 100% |
| 1b +root_y 通道 | -0.074（与臂1逐值相同） | 36% | 24.5 | 44% | 100% |
| 2 fullbody 自举 | **+0.000** | **100%** | 1.4 | **92%** | 100% |

**实测结论（E1+E2 后，E2 修正了 E1 结论 1）**：
1. 孤立 root_y 通道**分布级惰性**：noh 与 +root_y 两臂在全部 25 个 anchor 上逐值相同；
2. root-(x,z,yaw) **不是**充分控制：(x,z,yaw) 通道稳健（xz 100%<10cm，中位 3.9cm），但高度是解码器自选模态（y_end 散布 0.092–0.814，地板坐/椅坐随机切换；y 误差随座高增大：低/中/高三分位中位 -0.026/-0.083/-0.263）。E1 的"姿态载体携带高度基本够用"是单点运气；
3. 自举两步法（root-only 生成 → 终帧平移至 `座面高+0.17` → fullbody 终帧约束）**分布级成立**：25/25 终态高度误差 0.000、92% 零穿透；仅有的两例非零穿透（s186 帧152–212、s31 帧172–174）均为下坐瞬态且终态干净——正是测试时 GN 场景投影要修的残余，不是控制接口失败；
4. 自研 flow 设计输入加强版：root-y 必须是原生条件通道（含独立 dropout 模式），(x,y,z,yaw) 才能成为最小接口，省掉两步自举。

高度公式（construct() 用）：`pelvis_target = seat_h + 0.17`（按 decoder 自身身体标定；
不要用跨数据集 transl 差标定——会混入身高差，已踩过坑）。

**M2 construct() v1（完成，2026-08-12）**：解析几何 sit 候选生成 `o2a/construct.py` +
验收 `scripts/m2_eval.py`（静态物体 110 组 / 300 GT 模式，世界系聚类 0.25m/30°——
注意 anchor 的"物体局部系"实为支撑区域局部系，同物体不同坐点 origin 不同，不能直接比）。
结果：**候选集召回达标**（hit@all 87%，coverage@all 74%，零候选 7/110 组），
**排序是短板**（hit@1 20%，部分是多座点家具"有效但未采样"座位的指标假象）。
诊断与迭代记录见 `out/M2_summary.md`。结论：排序交给路线第②步非参数 anchor 库，
几何 construct() 定位为召回 + 硬过滤。

## 待办（按序）

1. **M2 第②步：非参数 anchor 库重排**：GT anchor 按物体局部系建库（注意支撑区域系问题，建议以候选座点为原点、候选 yaw 为朝向的局部特征：座高/靠背/开阔度/前缘距离 → 近邻打分），对 construct() 候选重排，验收指标同 m2_eval（目标 hit@5 ≥ 70%）；
2. M2 收尾杂项：7 组零候选（面片碎化）、桌前长凳朝向过杀（1984）、movable 三把椅子的逐帧物体系验收、E2 闭环接 construct() 候选（GT-free 全链路，即论文主实验雏形）；
3. **M1 标签清理**：7 条手持物误绑 sit anchor（movable:pen_01×2、book_left_01×4、cup_01×1，E2 选择时已临时过滤 batch_e2.py:select_anchors）——绑定谓词补"支撑面积/可坐性"过滤后重导 anchors.jsonl；
4. lie 谓词细化 + R 类（touch / pick 伸手时刻）；
5. 自研 flow 训练设计文档（表示、yaw 通道、预算 dropout、anchor 噪声增广）；
6. 审稿 agent 遗留行动项：残差外指标、检索+对齐 baseline、学习式条件化第四臂、‖E1−E0‖ 元验证、ShapeNet 物体级 OOD。

## 环境备忘（新会话必读）

- **绝不写 /home**（配额小）；一切缓存 `HF_HOME=/scratch/pf2m24/data/hf_cache`、`PIP_CACHE_DIR=/scratch/pf2m24/.cache/pip`；GB 级下载先问用户；
- 计算节点无外网：登录节点预缓存，运行时 `LOCAL_CACHE=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；
- GPU 用法：`squeue -lu pf2m24` 找交互 job（名 my_inter，a100 分区），`srun --jobid=<id> --overlap --export=ALL bash -c "..."`；
- conda 环境：`kimodo`（生成+可视化全套）、`motionbricks`（遗留，提取脚本也能跑）；
- Kimodo 输出 `root_positions` = SMPL-X 骨盆关节（≠ TRUMANS 的 transl，差 ~0.26m 且含体型差，勿直接比较）；
- 相关代码库：kimodo → /scratch/pf2m24/projects/kimodo；SFControl（自研 GN 机制出处）→ /scratch/pf2m24/projects/Umdd/SFControl；MotionBricks（仅 related work）→ /scratch/pf2m24/projects/GR00T-WholeBodyControl。

## 文件地图

```
o2a/                taxonomy.py(动词分类) trumans.py(shard读取) support.py(S类谓词)
                    construct.py(M2 解析几何 sit 候选生成)
scripts/            extract_anchors.py  cluster_viz.py  gen_kimodo.py(CLI版)
                    gen_kimodo2.py(进程内, --no-height/--fullbody)  check_gen.py
                    scene_check.py(场景合规, 仅movable)  viz3d.py(3D渲染)
                    batch_e2.py(批量三臂, --dry/--limit, 可断点续跑)
                    e2_metrics.py(合规指标, movable+静态, 局部座面)
                    m2_eval.py(construct 验收: hit@k/coverage, --thresh)
out/anchors.jsonl   876 条 GT anchor
out/viz3d/          E1_report.html(汇总报告) + 三臂 GIF/strip
out/gen_sit_0*.npz  E1 三臂生成结果
out/e2/             E2: gen_s{idx}_{noh|h|fb}.npz ×75 + e2_anchors.json(选择清单)
                    + e2_results.jsonl(逐条指标) + E2_summary.md(汇总)
```
