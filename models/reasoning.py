import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultimodalReasoning(nn.Module):
    def __init__(self, node_feat_dim=256, text_feat_dim=3072, device='cuda'):
        super().__init__()
        self.device = device
        
        # 1. 初始化 Phi-3-mini (用于指令解析)
        # 注意：这里我们只使用其 Transformer 层提取特征
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        self.text_encoder = AutoModel.from_pretrained("microsoft/Phi-3-mini-4k-instruct", 
                                                    trust_remote_code=True).to(device)
        
        # 2. 跨模态融合模块 (Cross-Attention)
        # Query 来自图节点，Key/Value 来自文本特征
        self.cross_attn = nn.MultiheadAttention(embed_dim=node_feat_dim, num_heads=8, batch_first=True)
        
        # 文本特征投影到节点特征维度
        self.text_proj = nn.Linear(text_feat_dim, node_feat_dim)
        
        # 3. 最终预测头 (预测位移)
        self.regressor = nn.Sequential(
            nn.Linear(node_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # 预测 (dx, dy, dz)
        )

    def get_text_embeddings(self, text):
        """
        获取文本指令的特征向量
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # 使用最后层的所有 token embedding 作为 key/value
            text_feats = outputs.last_hidden_state # [B, L, 3072]
        return text_feats

    def forward(self, node_features, text_query):
        """
        node_features: [N, D] (经 GNN 更新后的特征)
        text_query: str
        """
        # 1. 文本编码
        text_feats = self.get_text_embeddings(text_query) # [1, L, 3072]
        text_feats = self.text_proj(text_feats) # [1, L, D]
        
        # 2. 跨模态注意力融合
        # node_features 增加 batch 维度: [1, N, D]
        node_features = node_features.unsqueeze(0)
        
        # x_fused: [1, N, D]
        x_fused, _ = self.cross_attn(node_features, text_feats, text_feats)
        
        # 3. 位移预测
        # 去掉 batch 维度: [N, D] -> [N, 3]
        prediction = self.regressor(x_fused.squeeze(0))
        
        return prediction
