import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GNNReasoningLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        # 使用 GATv2，它在动态图和更复杂的交互中表现更好
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        """
        x: 节点特征 [N, in_channels]
        edge_index: 边索引 [2, E]
        """
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        return x

class GraphReasoningEngine(nn.Module):
    def __init__(self, node_feat_dim=384, hidden_dim=256):
        super().__init__()
        # 384 是 DINOv2-ViT-S 的特征维度
        self.gnn = GNNReasoningLayer(node_feat_dim, hidden_dim, hidden_dim)
        
    def forward(self, node_features, adjacency):
        """
        node_features: [N, D]
        adjacency: [N, N] 邻接矩阵
        """
        # 将邻接矩阵转换为 PyG 的 edge_index 格式
        edge_index = adjacency.nonzero().t().contiguous()
        
        if edge_index.numel() == 0:
            # 如果没有边，返回原始特征经过线性变换
            return node_features
            
        updated_features = self.gnn(node_features, edge_index)
        return updated_features
