import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import sys
sys.path.append("TextBoxesPP")

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw

# --------- config you should change ----------
img_dir = "/media/dongli911/Documents/Datasets/ICDAR2015/imgs/test"
out_dir = "TextBoxesPP/output"
weight_path = "AllConfig/all_model/ICDAR2013_TextBoxes.pth"
# input_size = 600  # keep your original behavior
max_images = 100 # set to e.g. 10 for quick test, or None for all
# --------------------------------------------

os.makedirs(out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

print("Loading model..")
net = RetinaNet().to(device)

# load weights (compatible with both formats)
ckpt = torch.load(weight_path, map_location="cpu")
if isinstance(ckpt, dict) and "net" in ckpt:
    net.load_state_dict(ckpt["net"])
else:
    net.load_state_dict(ckpt)

net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

encoder = DataEncoder()

val_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))]

for n, fname in enumerate(sorted(val_list)):
    print("Loading image...", fname)
    img_path = os.path.join(img_dir, fname)
    img = Image.open(img_path).convert("RGB")

    # w = h = input_size
    # img_resized = img.resize((w, h))

    print("Predicting..")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        loc_preds, cls_preds = net(x)

    print("Decoding..")
    boxes, labels, scores = encoder.decode(
        loc_preds.squeeze(0),
        cls_preds.squeeze(0),
        img.size
    )

    # draw
    draw = ImageDraw.Draw(img)
    boxes_np = boxes.reshape(-1, 4, 2)

    for box in boxes_np:
        draw.polygon(np.expand_dims(box, 0), outline=(0, 255, 0))

    out_path = os.path.join(out_dir, fname)
    img.save(out_path)

    if max_images is not None and (n + 1) >= max_images:
        break

print("Done.")
