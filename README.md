# PreAnchor (object2anchor)

用数学方法求解稀疏控制帧（root anchor），驱动冻结的、scene-blind 的动作生成器
完成人-场景交互；测试时用 GN 场景投影做小幅修正。核心科学问题：**场景信息需要
多少、通过哪条通道进入？**

- 权威进度文档：[STATUS.md](STATUS.md)（决策记录 / 里程碑结果 / 待办 / 环境备忘）
- 实验汇总：`out/E2_summary.md`（三臂控制接口验证）、`out/M2_summary.md`（解析几何求解器验收）

```
o2a/       核心库：taxonomy(动词→S/R/T) trumans(数据读取) support(S类谓词) construct(sit候选求解)
scripts/   提取 / 生成 / 批量实验 / 指标 / 可视化
out/       anchors.jsonl(GT标签) + 实验结果(jsonl/md；npz与图片不入库)
```

数据：TRUMANS（zirobtc 镜像）。生成器：Kimodo-SMPLX-RP-v1（冻结基线）。
