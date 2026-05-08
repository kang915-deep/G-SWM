import torch
import torch.nn as nn
from .graph_builder import SceneGraphBuilder
from .gnn_module import GraphReasoningEngine
from .reasoning import MultimodalReasoning

class GSWM(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # 1. 构图模块 (Grounding DINO + DINOv2)
        self.graph_builder = SceneGraphBuilder(device=device)
        
        # 2. 图推理模块 (GNN)
        self.gnn_engine = GraphReasoningEngine(node_feat_dim=384, hidden_dim=256)
        
        # 3. 多模态融合与预测模块 (Phi-3 + Cross-Attention)
        self.multimodal_reasoner = MultimodalReasoning(node_feat_dim=256, device=device)

    def forward(self, image, text_query):
        """
        image: PIL Image or Tensor
        text_query: str
        """
        # 1. 构建图
        graph_data = self.graph_builder(image, text_query)
        
        if graph_data is None:
            return None # 未发现物体
            
        node_features = graph_data["node_features"]
        adjacency = graph_data["adjacency"]
        
        # 2. 图神经推理 (捕捉物体间物理约束)
        updated_node_feats = self.gnn_engine(node_features, adjacency)
        
        # 3. 结合文本指令进行位移预测
        predictions = self.multimodal_reasoner(updated_node_feats, text_query)
        
        return {
            "pred_displacements": predictions, # [N, 3]
            "boxes": graph_data["boxes"],
            "labels": graph_data["labels"]
        }
