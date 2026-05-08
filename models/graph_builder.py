import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch.nn.functional as F

class SceneGraphBuilder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # 1. 初始化 Grounding DINO (用于物体检测)
        # 注意：实际运行时需要确保远程环境已下载模型
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)
        
        # 2. 初始化 DINOv2 (用于特征提取)
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        self.dinov2.eval()

    @torch.no_grad()
    def get_detections(self, image, text_queries):
        """
        使用 Grounding DINO 获取物体框
        image: PIL Image or Tensor
        text_queries: str, e.g., "red block. blue tray."
        """
        inputs = self.processor(images=image, text=text_queries, return_tensors="pt").to(self.device)
        outputs = self.grounding_dino(**inputs)
        
        # 处理输出获取 boxes 和 logits
        target_sizes = torch.tensor([image.size[::-1]]) if not isinstance(image, torch.Tensor) else torch.tensor([image.shape[-2:]])
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=target_sizes.to(self.device)
        )[0]
        
        return results["boxes"], results["scores"], results["labels"]

    def extract_node_features(self, image, boxes):
        """
        使用 DINOv2 提取每个 BBox 的特征
        """
        # 简化版实现：对原图进行 DINOv2 推理，然后根据 ROI Pooling 或直接裁剪获取特征
        # 这里为了演示，采用对每个 crop 进行特征提取
        node_features = []
        for box in boxes:
            x1, y1, x2, y2 = box.int()
            # 裁剪并缩放到 DINOv2 期望的大小 (通常是 224x224)
            crop = image[:, :, y1:y2, x1:x2]
            crop = F.interpolate(crop, size=(224, 224), mode='bilinear', align_corners=False)
            
            feat = self.dinov2(crop) # [1, 384] for vit-s
            node_features.append(feat)
            
        return torch.cat(node_features, dim=0) if node_features else torch.empty(0, 384).to(self.device)

    def build_adjacency_matrix(self, boxes, threshold=500.0):
        """
        根据物体中心点欧式距离构建邻接矩阵
        """
        if len(boxes) == 0:
            return torch.empty(0, 0).to(self.device)
            
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        dist_matrix = torch.cdist(centers, centers, p=2)
        
        # 阈值法构图
        adj = (dist_matrix < threshold).float()
        return adj

    def forward(self, image, text_queries):
        """
        端到端构建图
        """
        # 1. 检测物体
        boxes, scores, labels = self.get_detections(image, text_queries)
        
        # 2. 提取节点特征
        if boxes.shape[0] == 0:
            return None # 未检测到物体
            
        node_feats = self.extract_node_features(image, boxes)
        
        # 3. 构建邻接矩阵
        adj = self.build_adjacency_matrix(boxes)
        
        return {
            "node_features": node_feats,
            "adjacency": adj,
            "boxes": boxes,
            "labels": labels
        }
