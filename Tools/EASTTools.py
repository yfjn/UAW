import os
# ====== 动态加载 EAST repo（不强依赖当前项目结构）======
import importlib
import sys
import time

import cv2
import numpy as np
import torch

EAST_ROOT = "/media/dongli911/Documents/Workflow/YanFangjun/UAW/EAST"
EAST_CKPT = "AllConfig/all_model/east_best_model.pth.tar"

# EAST 相关全局句柄（避免重复 import）
_east_inited = False
_east_restore_rectangle = None
_east_polygon_area = None
_east_lanms = None
_east_resize_image = None


# ====== EAST postprocess/loader utils =======
def _east_init_imports():
    """
    动态把 EAST repo 加进来，并拿到：
    - restore_rectangle / polygon_area （EAST 的 data_utils.py）
    - lanms（EAST 的 lanms/）
    """
    global _east_inited, _east_restore_rectangle, _east_polygon_area, _east_lanms, _east_resize_image
    if _east_inited:
        return

    if not os.path.isdir(EAST_ROOT):
        raise RuntimeError(f"[EAST] EAST_ROOT not found: {EAST_ROOT}. Please set env EAST_ROOT or edit the script.")

    # 把 EAST 根目录放到 sys.path，使得可以 import EAST 的 model / data_utils / lanms
    if EAST_ROOT not in sys.path:
        sys.path.insert(0, EAST_ROOT)  # 使用 insert(0) 优先级更高

    # 直接实现 restore_rectangle 和 polygon_area，避免导入依赖Python2.7的data_utils.py
    def restore_rectangle_rbox(origin, geometry):
        d = geometry[:, :4]
        angle = geometry[:, 4]
        # for angle > 0
        origin_0 = origin[angle >= 0]
        d_0 = d[angle >= 0]
        angle_0 = angle[angle >= 0]
        if origin_0.shape[0] > 0:
            p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                          d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                          d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                          np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                          d_0[:, 3], -d_0[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_0 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_0 = np.zeros((0, 4, 2))
        # for angle < 0
        origin_1 = origin[angle < 0]
        d_1 = d[angle < 0]
        angle_1 = angle[angle < 0]
        if origin_1.shape[0] > 0:
            p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                          np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                          np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                          -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                          -d_1[:, 1], -d_1[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_1 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_1 = np.zeros((0, 4, 2))
        return np.concatenate([new_p_0, new_p_1])

    def polygon_area(poly):
        """
        compute area of a polygon
        :param poly:
        :return:
        """
        poly_ = np.array(poly)
        assert poly_.shape == (4,2), 'poly shape should be 4,2'
        edge = [
            (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
        ]
        return np.sum(edge)/2.

    _east_restore_rectangle = restore_rectangle_rbox
    _east_polygon_area = polygon_area

    try:
        # 强制重新加载，避免缓存问题
        if "lanms" in sys.modules:
            del sys.modules["lanms"]
        _east_lanms = importlib.import_module("lanms")
    except Exception as e:
        raise RuntimeError(f"[EAST] Failed to import lanms from {EAST_ROOT}. Error: {e}")

    # resize_image：直接复用一份与 EAST eval.py 等价的实现（避免你 EAST repo 的 eval.py 路径不一致）
    def _resize_image(im, max_side_len=2400):
        """
        resize image to a size multiple of 32 which is required by the network
        :param im: the resized image (RGB ndarray)
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        """
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # 防止过小导致负数
        resize_h = max(32, resize_h)
        resize_w = max(32, resize_w)

        resize_h = resize_h if resize_h % 32 == 0 else max(32, (resize_h // 32) * 32)
        resize_w = resize_w if resize_w % 32 == 0 else max(32, (resize_w // 32) * 32)

        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    _east_resize_image = _resize_image
    _east_inited = True


def _east_detect(score_map, geo_map, timer, score_map_thresh=1e-5, box_thresh=1e-8, nms_thres=0.1):
    """
    基于 EAST 常见实现的 decode：
    - restore_rectangle + lanms.merge_quadrangle_n9
    - 输出 boxes: (N,9) 前 8 个是四点坐标，最后一个是 score
    """
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    xy_text = np.argwhere(score_map > score_map_thresh)
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    start = time.time()
    text_box_restored = _east_restore_rectangle(
        xy_text[:, ::-1] * 4,
        geo_map[xy_text[:, 0], xy_text[:, 1], :]
    )  # N*4*2

    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start

    start = time.time()
    boxes = _east_lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start
    if boxes.shape[0] == 0:
        return None, timer

    # average score filter
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]

    boxes = boxes[boxes[:, 8] > box_thresh]
    if boxes.shape[0] == 0:
        return None, timer

    return boxes, timer


def _east_sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def east_tensor_to_boxes(east_model, img_tensor_bchw):
    """
    给定 EAST 模型 + 输入 tensor(B,3,H,W)，返回 boxes (N,4,2) 的 numpy
    """
    _east_init_imports()

    device = next(east_model.parameters()).device
    if img_tensor_bchw.device != device:
        img_tensor_bchw = img_tensor_bchw.to(device)

    # tensor -> rgb ndarray (H,W,3) uint8
    # 你 DataLoader 的 ToTensor 输出是 0~1，这里转回 0~255
    img_np = img_tensor_bchw[0].detach().float().cpu().numpy()  # 3,H,W
    img_np = np.transpose(img_np, (1, 2, 0))  # H,W,3
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    im_resized, (ratio_h, ratio_w) = _east_resize_image(img_np)
    im_resized = im_resized.astype(np.float32)
    im_t = torch.from_numpy(im_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    timer = {'net': 0, 'restore': 0, 'nms': 0}
    start = time.time()
    with torch.no_grad():
        score, geometry = east_model(im_t)
    timer['net'] = time.time() - start

    score = score.permute(0, 2, 3, 1).detach().cpu().numpy()
    geometry = geometry.permute(0, 2, 3, 1).detach().cpu().numpy()

    boxes, timer = _east_detect(score_map=score, geo_map=geometry, timer=timer)
    if boxes is None:
        return np.zeros((0, 4, 2), dtype=np.float32)

    boxes = boxes[:, :8].reshape((-1, 4, 2))
    boxes[:, :, 0] /= ratio_w
    boxes[:, :, 1] /= ratio_h

    # 过滤一下异常小框/负值
    final_boxes = []
    for box in boxes:
        box = _east_sort_poly(box.astype(np.int32))
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        if (box < 0).any():
            continue

        poly = np.array([[box[0, 0], box[0, 1]],
                         [box[1, 0], box[1, 1]],
                         [box[2, 0], box[2, 1]],
                         [box[3, 0], box[3, 1]]])
        p_area = _east_polygon_area(poly)
        if p_area > 0:
            poly = poly[(0, 3, 2, 1), :]
        final_boxes.append(poly)

    if len(final_boxes) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    return np.array(final_boxes, dtype=np.float32)


def load_east_model(weight_path=EAST_CKPT, device="cuda:0"):
    """
    加载 EAST repo 下的 model.py 网络，并加载权重
    """
    _east_init_imports()
    if not os.path.isfile(weight_path):
        raise RuntimeError(f"[EAST] checkpoint not found: {weight_path}. Please set env EAST_CKPT or edit the script.")

    east_model_mod = importlib.import_module("model")

    # 尝试自动找到网络类名（不同 fork 可能不一样）
    candidate_cls = [
        "EAST", "East", "EASTModel", "EASTNet", "EAST_network", "Network", "Model"
    ]
    NetCls = None
    for name in candidate_cls:
        if hasattr(east_model_mod, name):
            NetCls = getattr(east_model_mod, name)
            break
    if NetCls is None:
        raise RuntimeError(
            f"[EAST] Cannot find EAST network class in {EAST_ROOT}/model.py. "
            f"Please open model.py and add its class name into candidate_cls list."
        )

    net = NetCls()
    net = net.to(device)

    ckpt = torch.load(weight_path, map_location=device)
    # 兼容：直接是 state_dict 或 dict 包着
    if isinstance(ckpt, dict) and ("state_dict" in ckpt):
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # 兼容 DataParallel 的 module. 前缀
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v

    net.load_state_dict(new_state, strict=False)
    net.eval()
    return net