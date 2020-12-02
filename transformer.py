import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import YOLOLayer
from utils.utils import build_targets, to_cpu, non_max_suppression

def scaled_anchors(img_dim):
    anchors = [(10, 13), (16, 30), (33, 23)] + [(30, 61), (62, 45), (59, 119)] + [(116, 90), (156, 198), (373, 326)]
    new_anchors = []
    for x,y in anchors:
        new_anchors.append((int(x*img_dim/416), int(y*img_dim/416)))
    return [new_anchors[:3], new_anchors[3:6], new_anchors[6:]]

class Vision_Transformer(nn.Module):
    def __init__(self, num_classes =80, backbone = 'vit_base_patch16_384', pretrained = True, img_dim = 384):
        super(Vision_Transformer, self).__init__()
        self.transformer = timm.create_model(backbone, pretrained=pretrained)
        self.mappings = nn.ModuleList([nn.Conv2d(768,255,1), nn.Conv2d(768,255,1), nn.Conv2d(768,255,1)])
        self.img_dim = img_dim
        anchors = scaled_anchors(img_dim)
        self.features_layers = (7, 9, 11)
        self.yolo_layers = nn.ModuleList([YOLOLayer(anchors[0], num_classes, img_dim), 
                            YOLOLayer(anchors[1], num_classes, img_dim), 
                            YOLOLayer(anchors[2], num_classes, img_dim)])
        self.seen = 0
    
    def transformer_features(self, x):
        B = x.shape[0]
        x = self.transformer.patch_embed(x)

        cls_tokens = self.transformer.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + F.upsample(self.transformer.pos_embed.unsqueeze(0), size = (x.shape[1], x.shape[2])).squeeze(0)
        x = self.transformer.pos_drop(x)
        outputs = []
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x)
            if i in self.features_layers:
                out = x.permute(0,2,1)
                patch_dim = int(np.sqrt(out.shape[2] - 1))
                out = out[:,:,1:].view((out.shape[0],out.shape[1], patch_dim, patch_dim))
                outputs.append(out)
        return outputs
    
    def forward(self, x, targets=None, img_dim=None):
        features = self.transformer_features(x)
        mapped_features = [layer(inp) for layer, inp in zip(self.mappings, features)]
        if img_dim is None:
            img_dim = self.img_dim
        yolo_outputs = []
        loss = 0
        for layer,inp in zip(self.yolo_layers, mapped_features):
            layer_output, layer_loss = layer(inp, targets, img_dim)
            yolo_outputs.append(layer_output)
            loss += layer_loss
        yolo_outputs =  to_cpu(torch.cat(yolo_outputs, 1))
        return loss, yolo_outputs
        
     
