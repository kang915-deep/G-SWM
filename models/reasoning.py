import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultimodalReasoning(nn.Module):
    def __init__(self, node_feat_dim=256, text_feat_dim=3072, device='cuda'):
        super().__init__()
        self.device = device
        
        # 1. 初始化 Phi-3-mini
        # 显存优化：强制使用 float16 加载，V100 必备
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        self.text_encoder = AutoModel.from_pretrained("microsoft/Phi-3-mini-4k-instruct", 
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.float16).to(device)
        
        # 2. 跨模态融合模块
        self.cross_attn = nn.MultiheadAttention(embed_dim=node_feat_dim, num_heads=8, batch_first=True)
        self.text_proj = nn.Linear(text_feat_dim, node_feat_dim)
        
        # 3. 最终预测头 (由 regressor 改名为 predictor)
        self.predictor = nn.Sequential(
            nn.Linear(node_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3) 
        ).to(device)

    def get_text_embeddings(self, text):
        """
        获取文本指令的特征向量
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # 使用最后层 token embedding，并转换为 float32 匹配投影层
            text_feats = outputs.last_hidden_state.to(torch.float32)
        return text_feats

    def forward(self, node_features, text_query):
        # 1. 文本编码
        text_feats = self.get_text_embeddings(text_query) 
        text_feats = self.text_proj(text_feats)
        
        # 2. 跨模态注意力融合
        node_features = node_features.unsqueeze(0)
        x_fused, _ = self.cross_attn(node_features, text_feats, text_feats)
        
        # 3. 位移预测
        prediction = self.predictor(x_fused.squeeze(0))
        
        return prediction
