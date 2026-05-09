# G-SWM (Graph-Augmented Simple World Model)

G-SWM 是一个专为具身智能（Embodied AI）设计的结构化世界模型。它通过将非结构化的视觉输入转化为**动态场景图 (Dynamic Scene Graphs)**，并结合图神经网络（GNN）与轻量化大语言模型（Phi-3），实现了对复杂物理交互的高精度预测。

---

## 🌟 核心特性

- **结构化先验**：利用 Scene Graphs 显式建模物体间的空间拓扑关系，显著提升物理约束的理解力。
- **多模态融合**：整合 Grounding DINO (目标发现)、DINOv2 (视觉特征) 与 Phi-3 (语义推理)。
- **V100 深度优化**：针对 NVIDIA V100 (32GB) 集群进行了特化的混合精度与显存优化。
- **工业级分布式支持**：原生支持 PyTorch DDP (Distributed Data Parallel)，适配多机多卡训练。

---

## 🏗️ 系统架构

1.  **Scene Graph Construction** (`models/graph_builder.py`): 
    - 使用 Grounding DINO 自动提取物体 BBox。
    - 使用 DINOv2 提取 ROI 特征作为图节点。
2.  **Physical Interaction Engine** (`models/gnn_module.py`):
    - 基于 GATv2 层实现节点间的消息传递，捕捉物体间的物理相互作用力。
3.  **Multimodal Reasoning Head** (`models/reasoning.py`):
    - 引入 Phi-3-mini 进行指令解析。
    - 通过 Cross-Attention 将文本语义注入图节点特征。

---

## 🚀 快速开始

### 1. 环境准备
```bash
conda create -n gswm python=3.10 -y
conda activate gswm
pip install -r requirements.txt
```

### 2. 硬件特化配置 (针对 V100 8-GPU 节点)
本项目针对共享 GPU 资源和 NCCL 通讯进行了加固。在启动前，请务必设置以下环境变量：

```bash
# 解决 NCCL P2P 通讯挂起问题
export NCCL_P2P_DISABLE=1

# 指定 Python 路径
export PYTHONPATH=$PYTHONPATH:.
```

### 3. 启动分布式训练
```bash
torchrun --nproc_per_node=8 train_ddp.py \
    --epochs 10 \
    --batch_size 1 \
    --lr 1e-4
```

---

## 🛠️ 关键技术细节 (针对开发者)

### 显存优化策略
- **混合精度加载**：Phi-3 模型强制使用 `float16` 模式加载，单卡显存占用减少约 40%。
- **动态超时**：DDP 初始化超时时间已手动延长至 1 小时，以应对高负载环境下的计算延迟。

### 鲁棒性保护
- **检测加固**：内置坐标 Clamp 逻辑，自动修复 Grounding DINO 推理过程中的浮点越界问题。
- **零尺寸保护**：自动过滤并补全极小尺寸（<1px）的物体裁剪块。

---

## 📅 待办事项 (Roadmap)
- [ ] 集成 RLBench/RoboCas 真实数据集。
- [ ] 引入时间序列 Transformer 以支持多帧预测。
- [ ] 支持 WandB 实时训练监控。

---

## 📄 引用
如果你在研究中使用了本项目，请引用：
```bibtex
@software{G-SWM_2026,
  author = {Your Name},
  title = {G-SWM: Graph-Augmented Simple World Model},
  year = {2026},
  url = {https://github.com/your-username/G-SWM}
}
```
