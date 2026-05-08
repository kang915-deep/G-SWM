# G-SWM (Graph-Augmented Simple World Model)

本项目是一个简易的具身智能世界模型，通过引入**场景图 (Scene Graph)** 结构来增强对物理环境演变的预测能力。

## 环境要求
- Python 3.9+
- CUDA 11.x + 8x NVIDIA V100 (或同类 GPU)

## 远程服务器部署步骤

### 1. 克隆项目
```bash
git clone <你的GitHub仓库地址>
cd G-SWM
```

### 2. 创建环境并安装依赖
```bash
conda create -n gswm python=3.9 -y
conda activate gswm
pip install -r requirements.txt
```

### 3. 下载预训练模型
模型会自动通过 `transformers` 和 `torch.hub` 下载，但建议在联网良好的环境下运行。
- Grounding DINO: `IDEA-Research/grounding-dino-tiny`
- DINOv2: `dinov2_vits14`
- Phi-3: `microsoft/Phi-3-mini-4k-instruct`

### 4. 启动分布式训练 (8卡 V100)
使用 `torchrun` 启动：
```bash
torchrun --nproc_per_node=8 train_ddp.py --epochs 50 --batch_size 4 --lr 1e-4
```

## 核心架构说明
- `models/graph_builder.py`: 负责从原始图像中识别物体并构建拓扑图。
- `models/gnn_module.py`: 使用 GATv2 捕捉物体间的物理相互作用。
- `models/reasoning.py`: 使用 Phi-3 处理文本指令并进行跨模态特征融合。
- `train_ddp.py`: 实现多卡数据并行训练。

## 注意事项
- 当前代码库默认包含 `DummyEmbodiedDataset` 用于冒烟测试。若要使用真实 RLBench 数据，请修改 `data/dataset.py`。
- 如果服务器无法直接访问 HuggingFace，请设置环境变量 `export HF_ENDPOINT=https://hf-mirror.com`。
