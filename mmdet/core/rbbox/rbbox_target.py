import torch
import numpy as np
import mmcv
from .transforms import thetaobb2delta, pointobb2delta, hobb2delta
from ..utils import multi_apply


def rbbox_target(pos_proposals_list,
                neg_proposals_list,
                pos_assigned_gt_inds_list, 
                gt_rbboxes_list,
                gt_labels_list,
                rbbox_test_cfg,
                target_means,
                target_stds,
                out_dim_reg=5,
                concat=True):
    labels, label_weights, rbbox_targets, rbbox_weights = multi_apply(
        rbbox_target_single,
        pos_proposals_list,
        neg_proposals_list,
        pos_assigned_gt_inds_list,
        gt_rbboxes_list,
        gt_labels_list,
        rbbox_test_cfg=rbbox_test_cfg,
        target_means=target_means,
        target_stds=target_stds,
        out_dim_reg=out_dim_reg)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        rbbox_targets = torch.cat(rbbox_targets, 0)
        rbbox_weights = torch.cat(rbbox_weights, 0)
    return labels, label_weights, rbbox_targets, rbbox_weights


def rbbox_target_single(pos_proposals,
                        neg_proposals,
                        pos_assigned_gt_inds, 
                        gt_rbboxes,
                        gt_labels,
                        rbbox_test_cfg,
                        target_means,
                        target_stds,
                        out_dim_reg=5):
    rbbox2delta = {"thetaobb": thetaobb2delta,
                    "pointobb": pointobb2delta,
                    "hobb": hobb2delta}
    num_pos = pos_proposals.size(0)
    num_neg = neg_proposals.size(0)
    num_samples = num_pos + num_neg
    labels = pos_proposals.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_proposals.new_zeros(num_samples)
    rbbox_targets = pos_proposals.new_zeros(num_samples, out_dim_reg)
    rbbox_weights = pos_proposals.new_zeros(num_samples, out_dim_reg)
    
    pos_gt_rbboxes = []
    pos_gt_labels = []
    if num_pos > 0:
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_rbboxe = gt_rbboxes[pos_assigned_gt_inds[i]]
            gt_label = gt_labels[pos_assigned_gt_inds[i]]

            pos_gt_rbboxes.append(gt_rbboxe.tolist())
            pos_gt_labels.append(gt_label.tolist())

        pos_weight = 1.0
        label_weights[:num_pos] = pos_weight
        rbbox_weights[:num_pos, :] = 1.0

        pos_gt_rbboxes = np.array(pos_gt_rbboxes)
        pos_gt_rbboxes = torch.from_numpy(np.stack(pos_gt_rbboxes)).float().to(pos_proposals.device)

        pos_gt_labels = np.array(pos_gt_labels)
        pos_gt_labels = torch.from_numpy(np.stack(pos_gt_labels)).float().to(pos_proposals.device)

        pos_rbbox_targets = rbbox2delta[rbbox_test_cfg.encode](pos_proposals, pos_gt_rbboxes, target_means, target_stds)
        rbbox_targets[:num_pos, :] = pos_rbbox_targets
        labels[:num_pos] = pos_gt_labels
    else:
        print("num_pos = 0")

    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, rbbox_targets, rbbox_weights