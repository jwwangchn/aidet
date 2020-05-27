import numpy as np
import cv2
import torch
import math
import pycocotools.mask as maskUtils

# ================== obb convert =======================

def pointobb2pointobb(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    return pointobb.tolist()

def keypoint2pointobb(keypoint):
    """
    convert keypoint to pointobb
        :param keypoint: list, (4, 3)
        return: (N, 8)
    """
    keypoint = np.int0(np.array(keypoint))
    keypoint = keypoint[:, 0:2]
    keypoint = np.resize(keypoint, (1, 8)).squeeze().tolist()
    return keypoint


def pointobb2thetaobb(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    pointobb = np.int0(np.array(pointobb))
    pointobb.resize(4, 2)
    rect = cv2.minAreaRect(pointobb)
    x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    theta = theta / 180.0 * np.pi
    thetaobb = [x, y, w, h, theta]
    
    return thetaobb

def thetaobb2pointobb(thetaobb):
    """
    docstring here
        :param self: 
        :param thetaobb: list, [x, y, w, h, theta]
    """
    box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4]*180.0/np.pi))
    box = np.reshape(box, [-1, ]).tolist()
    pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

    return pointobb

def pointobb2bbox(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
        return [xmin, ymin, xmax, ymax]
    """
    xmin = min(pointobb[0::2])
    ymin = min(pointobb[1::2])
    xmax = max(pointobb[0::2])
    ymax = max(pointobb[1::2])
    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox


def thetaobb2hobb(thetaobb, pointobb_sort_fun):
    """
    docstring here
        :param self: 
        :param thetaobb: list, [x, y, w, h, theta]
    """
    pointobb = thetaobb2pointobb(thetaobb)
    sorted_pointobb = pointobb_sort_fun(pointobb)
    first_point = [sorted_pointobb[0], sorted_pointobb[1]]
    second_point = [sorted_pointobb[2], sorted_pointobb[3]]

    end_point = [sorted_pointobb[6], sorted_pointobb[7]]
    
    h = np.sqrt((end_point[0] - first_point[0])**2 + (end_point[1] - first_point[1])**2)

    hobb = first_point + second_point + [h]
    
    return hobb


def pointobb_extreme_sort(pointobb):
    """
    Find the "top" point and sort all points as the "top right bottom left" order
        :param self: self
        :param points: unsorted points, (N*8) 
    """   
    points_np = np.array(pointobb)
    points_np.resize(4, 2)
    # sort by Y
    sorted_index = np.argsort(points_np[:, 1])
    points_sorted = points_np[sorted_index, :]
    if points_sorted[0, 1] == points_sorted[1, 1]:
        if points_sorted[0, 0] < points_sorted[1, 0]:
            sorted_top_idx = 0
        else:
            sorted_top_idx = 1
    else:
        sorted_top_idx = 0

    top_idx = sorted_index[sorted_top_idx]
    pointobb = pointobb[2*top_idx:] + pointobb[:2*top_idx]
    
    return pointobb


def pointobb_best_point_sort(pointobb):
    """
    Find the "best" point and sort all points as the order that best point is first point
        :param self: self
        :param points: unsorted points, (N*8) 
    """
    xmin, ymin, xmax, ymax = pointobb2bbox(pointobb)
    w = xmax - xmin
    h = ymax - ymin
    reference_bbox = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
    reference_bbox = np.array(reference_bbox)
    normalize = np.array([1.0, 1.0] * 4)
    combinate = [np.roll(pointobb, 0), np.roll(pointobb, 2), np.roll(pointobb, 4), np.roll(pointobb, 6)]
    distances = np.array([np.sum(((coord - reference_bbox) / normalize)**2) for coord in combinate])
    sorted = distances.argsort()

    return combinate[sorted[0]].tolist()


def hobb2pointobb(hobb):
    """
    docstring here
        :param self: 
        :param hobb: list, [x1, y1, x2, y2, h]
    """
    first_point_x = hobb[0]
    first_point_y = hobb[1]
    second_point_x = hobb[2]
    second_point_y = hobb[3]
    h = hobb[4]

    angle_first_second = np.pi / 2.0 - np.arctan2(second_point_y - first_point_y, second_point_x - first_point_x)
    delta_x = h * np.cos(angle_first_second)
    delta_y = h * np.sin(angle_first_second)

    forth_point_x = first_point_x - delta_x
    forth_point_y = first_point_y + delta_y

    third_point_x = second_point_x - delta_x
    third_point_y = second_point_y + delta_y

    pointobb = [first_point_x, first_point_y, second_point_x, second_point_y, third_point_x, third_point_y, forth_point_x, forth_point_y]

    pointobb = [int(_) for _ in pointobb]
    
    return pointobb


def maskobb2thetaobb(maskobb):
    mask = maskUtils.decode(maskobb).astype(np.bool)
    gray = np.array(mask*255, dtype=np.uint8)
    images, contours, hierarchy = cv2.findContours(gray.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if contours != []:
        imax_cnt_area = -1
        imax = -1
        for i, cnt in enumerate(contours):
            cnt_area = cv2.contourArea(cnt)
            if imax_cnt_area < cnt_area:
                imax = i
                imax_cnt_area = cnt_area
        cnt = contours[imax]
        rect = cv2.minAreaRect(cnt)
        x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
        thetaobb = [x, y, w, h, theta * np.pi / 180.0]
    else:
        thetaobb = [0, 0, 0, 0, 0]

    return thetaobb

# ================== flip obb =======================

def thetaobb_flip(thetaobbs, img_shape):
    """
    flip thetaobb
        :param self: 
        :param thetaobbs: np.array, [[x, y, w, h, theta]], (..., 5)
    """
    assert thetaobbs.shape[-1] % 5 == 0
    w = img_shape[1]
    flipped = thetaobbs.copy()
    flipped[..., 0] = w - flipped[..., 0] - 1
    flipped[..., [3, 2]] = flipped[..., [2, 3]]
    flipped[..., 4] = -math.pi/2.0 - flipped[..., 4]
    return flipped

def pointobb_flip(pointobbs, img_shape):
    """
    flip pointobbs
        :param self: 
        :param pointobbs: np.array, [[x1, y1, x2, y2, x3, y3, x4, y4]], (..., 8)
    """
    assert pointobbs.shape[-1] % 8 == 0
    pointobb_extreme_sort = False       # TODO: fix this when use the old sort method
    
    if pointobb_extreme_sort:
        w = img_shape[1]
        flipped = pointobbs.copy()
        flipped[..., 0::2] = w - flipped[..., 0::2] - 1
        flipped[..., [2, 6]] = flipped[..., [6, 2]]
        flipped[..., [3, 7]] = flipped[..., [7, 3]]
    else:
        w = img_shape[1]
        pointobbs_cp = pointobbs.copy()
        pointobbs_cp[..., 0::2] = w - pointobbs_cp[..., 0::2] - 1
        pointobbs_cp[..., [2, 6]] = pointobbs_cp[..., [6, 2]]
        pointobbs_cp[..., [3, 7]] = pointobbs_cp[..., [7, 3]]
        ndim_flag = False

        if pointobbs_cp.ndim == 1:
            ndim_flag = True
            pointobbs_cp = pointobbs_cp[np.newaxis, :]

        flipped = []
        for _ in pointobbs_cp:
            flipped.append(np.array(pointobb_best_point_sort(_.tolist())))

        flipped = np.array(flipped)
        if ndim_flag:
            flipped = flipped.squeeze()

    return flipped


def hobb_flip(hobbs, img_shape):
    """
    flip hobbs
        :param self: 
        :param hobbs: np.array, [[x1, y1, x2, y2, h]], (..., 5)
    """
    if hobbs.ndim == 1:
        hobbs = hobbs[np.newaxis, ...]
    assert hobbs.shape[-1] % 5 == 0
    w = img_shape[1]
    pointobbs = []
    for hobb in hobbs:
        pointobb = hobb2pointobb(hobb)
        pointobbs.append(pointobb)
    pointobbs = np.array(pointobbs)

    pointobb_extreme_sort = False       # TODO: fix this when use the old sort method
    
    if pointobb_extreme_sort:
        flipped = hobbs.copy()
        flipped[..., 4] = np.sqrt((flipped[..., 0] - flipped[..., 2])**2 + (flipped[..., 1] - flipped[..., 3])**2)
        flipped[..., 0] = w - flipped[..., 0] - 1
        flipped[..., 1] = flipped[..., 1]
        flipped[..., 2] = w - pointobbs[..., 6] - 1
        flipped[..., 3] = pointobbs[..., 7]
        flipped = flipped.squeeze()
    else:
        pointobbs = pointobb_flip(pointobbs, img_shape)
        thetaobbs = [pointobb2thetaobb(pointobb) for pointobb in pointobbs]
        hobbs = [thetaobb2hobb(thetaobb, pointobb_best_point_sort) for thetaobb in thetaobbs]
        flipped = np.array(hobbs)
        
    return flipped


# ================== rescale obb =======================

def thetaobb_rescale(thetaobbs, scale_factor, reverse_flag=False):
    """
    rescale thetaobb
        :param self: 
        :param thetaobb: np.array, [[x, y, w, h, theta]], (..., 5)
        :param reverse_flag: bool, if reverse_flag=Ture -> reverse rescale
    """
    thetaobbs_ = thetaobbs.clone()
    if reverse_flag == False:
        thetaobbs *= scale_factor
    else:
        thetaobbs /= scale_factor
    thetaobbs[..., 4::5] = thetaobbs_[..., 4::5]
    return thetaobbs

def pointobb_rescale(pointobbs, scale_factor, reverse_flag=False):
    """
    rescale pointobb
        :param self: 
        :param pointobb: np.array, [[x, y, w, h, theta]], (..., 5)
        :param reverse_flag: bool, if reverse_flag=Ture -> reverse rescale
    """
    if reverse_flag == False:
        pointobbs *= scale_factor
    else:
        pointobbs /= scale_factor
    return pointobbs

def hobb_rescale(hobbs, scale_factor, reverse_flag=False):
    """
    rescale hobb
        :param self: 
        :param hobb: np.array, [[x, y, w, h, theta]], (..., 5)
        :param reverse_flag: bool, if reverse_flag=Ture -> reverse rescale
    """
    if reverse_flag == False:
        hobbs *= scale_factor
    else:
        hobbs /= scale_factor
    return hobbs

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

def rbbox2result(rbboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        rbboxes (Tensor): shape (n, 6 or 9)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if rbboxes.shape[0] == 0:
        return [
            np.zeros((0, 6), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        rbboxes = rbboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [rbboxes[labels == i, :] for i in range(num_classes - 1)]