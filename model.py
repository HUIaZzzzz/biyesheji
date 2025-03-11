import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class ViTBase16(nn.Module):
    def __init__(self):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_base_patch16_384", pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, 5)
        state_dict = torch.load("./pytorch_model.bin",weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNetB4(nn.Module):
    def __init__(self):
        super(EfficientNetB4, self).__init__()
        # 先创建预训练模型但不包括分类器头部
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        # 然后添加我们自己的分类器
        state_dict = torch.load("./efnet.bin", weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.classifier = nn.Linear(self.model.num_features, 5)

    def forward(self, x):
        # 获取特征
        features = self.model(x)
        # 通过我们的分类器
        x = self.classifier(features)
        return x


class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.model1 = ViTBase16()
        self.model2 = EfficientNetB4()
        self.classifier = nn.Linear(10, 5)  # 5+5=10 features from both models

    def forward(self, x):
        # 调整输入尺寸
        x1 = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)  # ViT输入
        x2 = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)  # EfficientNet输入
        
        # Get predictions from both models
        out1 = self.model1(x1)
        out2 = self.model2(x2)
        
        # Concatenate the outputs
        combined = torch.cat((out1, out2), dim=1)
        
        # Final classification
        out = self.classifier(combined)
        return out
