import os
import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/UAW")
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/UAW/TextBoxesPP")
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from TextBoxesPP.retinanet import RetinaNet
from TextBoxesPP.encoder import DataEncoder

def load_TextBoxesPPmodel(weight_path, device="cuda:0"):
    """
    Load TextBoxes++ (RetinaNet-based) detector.

    weight_path can be:
      - a state_dict (.pth) OR
      - a checkpoint dict containing key 'net'
    """
    net = RetinaNet().to(device)
    ckpt = torch.load(weight_path, map_location="cpu")
    if isinstance(ckpt, dict) and "net" in ckpt:
        net.load_state_dict(ckpt["net"])
    else:
        net.load_state_dict(ckpt)
    net.eval()

    encoder = DataEncoder()
    return net, encoder

def _normalize_imagenet(x):
    """
    x: torch tensor [B,3,H,W], float, range [0,1]
    """
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std

def textboxespp_pred_tensor(net, encoder, img_tensor, input_size=None, cls_thresh=None, nms_thresh=None):
    """
    Run TextBoxes++ prediction on a tensor image.
    img_tensor: [1,3,H,W] on GPU (recommended), range [0,1]
    Return: boxes(list of np.ndarray Nx4x2), scores(list/np) - boxes are in original image coordinates
    """
    # Ensure input tensor is on the same device as the model
    device = next(net.parameters()).device
    if img_tensor.device != device:
        img_tensor = img_tensor.to(device)
    
    # Store original size for coordinate scaling
    orig_h, orig_w = img_tensor.shape[2], img_tensor.shape[3]
    
    # Resize to square like the original TextBoxes++ test script style
    if input_size:
        img_tensor = F.interpolate(img_tensor, size=(input_size, input_size), mode="bilinear", align_corners=False)
    
    # Normalize the image
    x = _normalize_imagenet(img_tensor)

    with torch.no_grad():
        loc_preds, cls_preds = net(x)

    # CRITICAL: The encoder.decode expects the input_size that matches what was fed to the network
    # Get the actual size that the network processed (after normalization, same as input)
    # Note: x.shape is [B, C, H, W], and encoder expects (W, H) tuple
    actual_w = x.shape[3]
    actual_h = x.shape[2]
    
    # DataEncoder.decode expects (width, height) tuple
    boxes, labels, scores = encoder.decode(
        loc_preds.squeeze(0),
        cls_preds.squeeze(0),
        (actual_w, actual_h)
    )

    # boxes is usually tensor shape [N, 8] (4 points) -> reshape to [N,4,2]
    if boxes.shape[0] == 0:
        return [], []
    boxes = boxes.reshape(-1, 4, 2)

    # scores
    if hasattr(scores, "detach"):
        scores = scores.detach()
    else:
        scores = np.array(scores, dtype=np.float32)

    # Optional: apply additional thresholding here if your encoder.decode doesn't already do it
    if cls_thresh is not None:
        keep = scores >= float(cls_thresh)
        boxes = boxes[keep]
        scores = scores[keep]

    # Scale boxes back to original image size if we resized
    if input_size and (orig_w != actual_w or orig_h != actual_h):
        scale_x = orig_w / actual_w
        scale_y = orig_h / actual_h
        # boxes shape: [N, 4, 2] where last dim is (x, y)
        for i in range(len(boxes)):
            boxes[i][:, 0] *= scale_x  # scale x coordinates
            boxes[i][:, 1] *= scale_y  # scale y coordinates

    # NOTE: nms_thresh usually applied inside encoder.decode; kept here for signature compatibility.
    return list(boxes), list(scores)