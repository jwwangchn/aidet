import mmcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair

from ..utils import multi_apply
from .transforms import offset2delta


def offset_target(pos_proposals_list, 
                 pos_assigned_gt_inds_list, 
                 gt_offsets_list,
                 cfg,
                 target_means=[.0, .0],
                 target_stds=[1.0, 1.0],
                 concat=True):
    offset_targets, _ = multi_apply(offset_target_single, 
                                 pos_proposals_list,
                                 pos_assigned_gt_inds_list, 
                                 gt_offsets_list,
                                 cfg=cfg,
                                 target_means=target_means,
                                 target_stds=target_stds)
    if concat:
        offset_targets = torch.cat(offset_targets, 0)
        
    return offset_targets


def offset_target_single(pos_proposals, 
                         pos_assigned_gt_inds, 
                         gt_offsets, 
                         cfg,
                         target_means=[.0, .0],
                         target_stds=[1.0, 1.0]):
    num_pos = pos_proposals.size(0)
    offset_targets = pos_proposals.new_zeros(pos_proposals.size(0), 2)

    pos_gt_offsets = []
    
    if num_pos > 0:
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_offset = gt_offsets[pos_assigned_gt_inds[i]]
            pos_gt_offsets.append(gt_offset.tolist())

        pos_gt_offsets = np.array(pos_gt_offsets)
        pos_gt_offsets = torch.from_numpy(np.stack(pos_gt_offsets)).float().to(pos_proposals.device)
        offset_targets = offset2delta(pos_proposals, pos_gt_offsets, means=target_means, stds=target_stds)
    else:
        print("num_pos = 0")

    return offset_targets, offset_targets
