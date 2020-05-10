import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, constant_init

import wwtool
from wwtool import ChannelAttention, SpatialAttention, show_featuremap

from mmdet.ops import ConvModule
from ..registry import HEADS


@HEADS.register_module
class WeightedPseudoSegmentationHead(nn.Module):
    def __init__(self,
                 num_convs=4,
                 in_channels=256,
                 inside_channels=128,
                 conv_out_channels=256,
                 num_classes=16,
                 ignore_label=255,
                 loss_weight=1.0,
                 with_background_reweight=False,
                 reweight_version='v1',
                 gamma=2,
                 alpha=0.5,
                 use_focal_loss=False,
                 conv_cfg=None,
                 norm_cfg=None):
        super(WeightedPseudoSegmentationHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.inside_channels = inside_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.with_background_reweight = with_background_reweight
        self.reweight_version = reweight_version
        self.gamma = gamma
        self.alpha = alpha
        self.use_focal_loss = use_focal_loss
        self.show_featuremap = False

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # Semantic Feature Extraction
        self.scale_heads = []
        self.in_features = ["p2", "p3", "p4", "p5"]
        self.feature_strides = {"p2":4, "p3":8, "p4":16, "p5":32, "p6":64}
        self.common_stride = 4
        self.norm = "GN"
        self.conv_dims = 128
        self.feature_channels = {"p2":256, "p3":256, "p4":256, "p5":256, "p6":256}
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(self.feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                # norm_module = nn.GroupNorm(32, self.conv_dims) if self.norm == "GN" else None
                conv = ConvModule(
                    self.feature_channels[in_feature] if k == 0 else self.conv_dims,
                    self.conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg
                )
                head_ops.append(conv)
                if self.feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # extra conv
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels_ = self.inside_channels if i == 0 else conv_out_channels
            self.convs.append(
                ConvModule(
                    in_channels_,
                    conv_out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        self.conv_embedding = ConvModule(
            self.conv_out_channels,
            conv_out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.conv_logits = nn.Conv2d(self.conv_out_channels, self.num_classes, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduce=False)

        # SENet
        self.reweight_version = reweight_version
        
        if self.reweight_version == 'v1':
            # v1
            self.fc1 = nn.Conv2d(self.conv_out_channels, self.conv_out_channels, kernel_size=1)
            self.conv_before_output = nn.Conv2d(self.conv_out_channels, self.conv_out_channels, kernel_size=3, stride=1, padding=1)
            self.gn3 = nn.GroupNorm(self.conv_out_channels, self.conv_out_channels)
        elif self.reweight_version == 'v2':
            # v2
            self.fc2 = nn.Conv2d(128, 64, kernel_size=1)
            self.fc3 = nn.Conv2d(64, 128, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, feats):
        # for idx, feat in enumerate(feats):
        #     print(idx, feat.shape)
        p6 = feats[4]
        p5 = feats[3]
        p4 = feats[2]
        p3 = feats[1]
        p2 = feats[0]

        features = {"p2":p2, "p3":p3, "p4":p4, "p5":p5, "p6":p6}
        
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])

        for i in range(self.num_convs):
            x = self.convs[i](x)

        # SENet
        if self.with_background_reweight:
            if self.reweight_version == 'v1':
                w = F.avg_pool2d(x, x.size(2))
                w = self.gn3(self.fc1(w))
                w = torch.sigmoid(w)
                x = x * w
                x = self.conv_before_output(x)

            elif self.reweight_version == 'v2':
                w = F.avg_pool2d(x, x.size(2))
                w = torch.relu(self.fc2(w))
                w = self.fc3(w)
                w = torch.sigmoid(w)
                x = x * w

        # x = self.upsample(x)

        mask_pred = self.conv_logits(x)
        x = self.conv_embedding(x)
        if self.show_featuremap:
            prediction = mask_pred.squeeze(0)
            prediction = F.softmax(prediction, dim=0).argmax(0).cpu().numpy()
            print("prediction: ", prediction[0:10, 0:10])
            wwtool.visualization.mask.show_mask(prediction, 16)
            show_featuremap(x.cpu(), win_name='SSB Output')
        return mask_pred, x

    def loss(self, mask_pred, labels, weights=None):
        if self.show_featuremap:
            show_featuremap(labels.cpu(),  win_name='labels')
            # show_featuremap(weights.cpu(),  win_name='weights')
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        if self.use_focal_loss:
            logpt = -loss_semantic_seg
            pt = torch.exp(logpt)
            loss_semantic_seg = -((1 - pt) ** self.gamma) * logpt
        if weights is not None:
            weights = weights.float()
            weights = (weights / 255.0 + 1.0) * self.loss_weight
            loss_semantic_seg = (loss_semantic_seg * weights.squeeze(1)).mean()
        else:
            loss_semantic_seg = loss_semantic_seg.mean()
        return loss_semantic_seg