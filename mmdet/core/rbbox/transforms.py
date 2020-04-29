import numpy as np
import cv2
import torch
import math

from mmdet.datasets.dota.transform import thetaobb_flip, pointobb_flip, hobb_flip

def thetaobb2delta(proposals, gt, means=[0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    # proposals: (x1, y1, x2, y2)
    # gt: (cx, cy, w, h, theta)
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0
    ptheta = np.ones(proposals.shape[0], dtype=np.int32) * (-np.pi/2.0)
    ptheta = torch.from_numpy(np.stack(ptheta)).float().to(proposals.device)

    gx = gt[..., 0]
    gy = gt[..., 1]
    gw = gt[..., 2] + 1.0
    gh = gt[..., 3] + 1.0
    gtheta = gt[..., 4]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    dtheta = gtheta - ptheta
    
    deltas = torch.stack([dx, dy, dw, dh, dtheta], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2thetaobb(rois,
                   deltas,
                   means=[0, 0, 0, 0, 0],
                   stds=[1, 1, 1, 1, 1],
                   max_shape=None,
                   wh_ratio_clip=16 / 1000):
    # print("deltas: ", deltas)
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    # print("denorm_deltas: ", denorm_deltas)
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dtheta = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))

    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    ptheta = np.ones(rois.shape[0], dtype=np.int32) * (-np.pi/2.0)
    ptheta = torch.from_numpy(np.stack(ptheta)).float().to(rois.device).unsqueeze(1).expand_as(dtheta)
    
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    gtheta = dtheta + ptheta

    if max_shape is not None:
        pass
    thetaobbs = torch.stack([gx, gy, gw, gh, gtheta], dim=-1).view_as(deltas)
    return thetaobbs


def thetaobb_mapping(thetaobbs, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_thetaobbs = thetaobbs * scale_factor
    if flip:
        new_thetaobbs = thetaobb_flip(new_thetaobbs, img_shape)
    return new_thetaobbs

def thetaobb_mapping_back(thetaobbs, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_thetaobbs = thetaobb_flip(thetaobbs, img_shape) if flip else thetaobbs
    new_thetaobbs = new_thetaobbs / scale_factor
    return new_thetaobbs


def pointobb2delta(proposals, gt, means=[0, 0, 0, 0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1, 1, 1, 1]):
    # proposals: (x1, y1, x2, y2)
    # gt: (x1, y1, x2, y2, x3, y3, x4, y4)
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0
    pxmin, pymin, pxmax, pymax = proposals[..., 0], proposals[..., 1], proposals[..., 2], proposals[..., 3]
    px1 = pxmin
    py1 = pymin
    px2 = pxmax
    py2 = pymin
    px3 = pxmax
    py3 = pymax
    px4 = pxmin
    py4 = pymax

    gx1 = gt[..., 0]
    gy1 = gt[..., 1]
    gx2 = gt[..., 2]
    gy2 = gt[..., 3]
    gx3 = gt[..., 4]
    gy3 = gt[..., 5]
    gx4 = gt[..., 6]
    gy4 = gt[..., 7]

    dx1 = (gx1 - px1) / pw
    dy1 = (gy1 - py1) / ph
    dx2 = (gx2 - px2) / pw
    dy2 = (gy2 - py2) / ph
    dx3 = (gx3 - px3) / pw
    dy3 = (gy3 - py3) / ph
    dx4 = (gx4 - px4) / pw
    dy4 = (gy4 - py4) / ph
    
    deltas = torch.stack([dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2pointobb(rois,
                   deltas,
                   means=[0, 0, 0, 0, 0, 0, 0, 0],
                   stds=[1, 1, 1, 1, 1, 1, 1, 1],
                   max_shape=None,
                   wh_ratio_clip=16 / 1000):
    # print("deltas: ", deltas)
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 8)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 8)
    denorm_deltas = deltas * stds + means

    # print("denorm_deltas: ", denorm_deltas)
    dx1 = denorm_deltas[:, 0::8]
    dy1 = denorm_deltas[:, 1::8]
    dx2 = denorm_deltas[:, 2::8]
    dy2 = denorm_deltas[:, 3::8]
    dx3 = denorm_deltas[:, 4::8]
    dy3 = denorm_deltas[:, 5::8]
    dx4 = denorm_deltas[:, 6::8]
    dy4 = denorm_deltas[:, 7::8]

    pxmin, pymin, pxmax, pymax = rois[..., 0], rois[..., 1], rois[..., 2], rois[..., 3]
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dx1)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dy1)
    
    px1 = pxmin.unsqueeze(1).expand_as(dx1)
    py1 = pymin.unsqueeze(1).expand_as(dy1)
    px2 = pxmax.unsqueeze(1).expand_as(dx2)
    py2 = pymin.unsqueeze(1).expand_as(dy2)
    px3 = pxmax.unsqueeze(1).expand_as(dx3)
    py3 = pymax.unsqueeze(1).expand_as(dy3)
    px4 = pxmin.unsqueeze(1).expand_as(dx4)
    py4 = pymax.unsqueeze(1).expand_as(dy4)

    gx1 = pw * dx1 + px1
    gy1 = ph * dy1 + py1
    gx2 = pw * dx2 + px2
    gy2 = ph * dy2 + py2
    gx3 = pw * dx3 + px3
    gy3 = ph * dy3 + py3
    gx4 = pw * dx4 + px4
    gy4 = ph * dy4 + py4

    if max_shape is not None:
        pass
    pointobbs = torch.stack([gx1, gy1, gx2, gy2, gx3, gy3, gx4, gy4], dim=-1).view_as(deltas)
    return pointobbs


def pointobb_mapping(pointobbs, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_pointobbs = pointobbs * scale_factor
    if flip:
        new_pointobbs = pointobb_flip(new_pointobbs, img_shape)
    return new_pointobbs

def pointobb_mapping_back(pointobbs, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_pointobbs = pointobb_flip(pointobbs, img_shape) if flip else pointobbs
    new_pointobbs = new_pointobbs / scale_factor
    return new_pointobbs



def hobb2delta(proposals, gt, means=[0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    # proposals: (x1, y1, x2, y2)
    # gt: (x1, y1, x2, y2, h)
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0
    pxmin, pymin, pxmax, pymax = proposals[..., 0], proposals[..., 1], proposals[..., 2], proposals[..., 3]
    px1 = pxmin
    py1 = pymin
    px2 = pxmax
    py2 = pymin
    ph = pymax - pymin + 1.0

    gx1 = gt[..., 0]
    gy1 = gt[..., 1]
    gx2 = gt[..., 2]
    gy2 = gt[..., 3]
    gh = gt[..., 4] + 1.0

    dx1 = (gx1 - px1) / pw
    dy1 = (gy1 - py1) / ph
    dx2 = (gx2 - px2) / pw
    dy2 = (gy2 - py2) / ph
    dh = (gh - ph) / ph
    
    deltas = torch.stack([dx1, dy1, dx2, dy2, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def delta2hobb(rois,
                deltas,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1],
                max_shape=None,
                wh_ratio_clip=16 / 1000):
    # print("deltas: ", deltas)
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    # print("denorm_deltas: ", denorm_deltas)
    dx1 = denorm_deltas[:, 0::5]
    dy1 = denorm_deltas[:, 1::5]
    dx2 = denorm_deltas[:, 2::5]
    dy2 = denorm_deltas[:, 3::5]
    dh = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))

    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    pxmin, pymin, pxmax, pymax = rois[..., 0], rois[..., 1], rois[..., 2], rois[..., 3]
    
    px1 = pxmin.unsqueeze(1).expand_as(dx1)
    py1 = pymin.unsqueeze(1).expand_as(dy1)
    px2 = pxmax.unsqueeze(1).expand_as(dx2)
    py2 = pymin.unsqueeze(1).expand_as(dy2)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)

    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dh)

    gx1 = pw * dx1 + px1
    gy1 = ph * dy1 + py1
    gx2 = pw * dx2 + px2
    gy2 = ph * dy2 + py2
    gh = ph * dh + ph

    if max_shape is not None:
        pass
    hobbs = torch.stack([gx1, gy1, gx2, gy2, gh], dim=-1).view_as(deltas)
    return hobbs


def hobb_mapping(hobbs, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_hobbs = hobbs * scale_factor
    if flip:
        new_hobbs = hobb_flip(new_hobbs, img_shape)
    return new_hobbs

def hobb_mapping_back(hobbs, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_hobbs = hobb_flip(hobbs, img_shape) if flip else hobbs
    new_hobbs = new_hobbs / scale_factor
    return new_hobbs