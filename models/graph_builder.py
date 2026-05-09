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
        """
        inputs = self.processor(images=image, text=text_queries, return_tensors="pt").to(self.device)
        outputs = self.grounding_dino(**inputs)
        
        # 获取图像尺寸用于后处理和 Clamp
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        else:
            width, height = image.size

        target_sizes = torch.tensor([[height, width]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            target_sizes=target_sizes
        )[0]
        
        boxes = results["boxes"]
        # 强制 Clamp 坐标，防止微小浮点误差导致越界 (非常重要)
        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=height)
        
        return boxes, results["scores"], results["labels"]

    def extract_node_features(self, image, boxes):
        """
        使用 DINOv2 提取每个 BBox 的特征
        """
        node_features = []
        _, h, w = image.shape if isinstance(image, torch.Tensor) else (3, image.size[1], image.size[0])
        
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            
            # 极端情况兜底：确保裁剪尺寸至少为 1 像素且在图像范围内
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1: x2 = x1 + 1
            if y2 <= y1: y2 = y1 + 1
            
            crop = image[:, y1:y2, x1:x2]
            
            # 如果裁剪后依然为空，补零
            if crop.numel() == 0:
                feat = torch.zeros((1, 384), device=self.device)
            else:
                crop_resized = F.interpolate(crop.unsqueeze(0), size=(224, 224), 
                                            mode='bilinear', align_corners=False)
                feat = self.dinov2(crop_resized)
                
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
