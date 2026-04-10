import os
import sys
import tempfile
from mmcv import Config
from mmcv.utils import import_modules_from_strings
# from mmocr.utils.ocr import MMOCR  # 延迟导入，避免启动时报错

# 说明：
# - 这里的实现严格对齐 ocrclip/test.py 的 TCM 测试逻辑：
#   1) 读取 config
#   2) 如果有 custom_imports，则先 import（否则会找不到自定义模块）
#   3) 将 cfg.model.pretrained 以及 neck.rfp_backbone.pretrained 置为 None（避免额外加载预训练权重）
#   4) 将“修补后的 config”写入临时文件，再用 MMOCR(det_config, det_ckpt) 初始化 detector-only 引擎
#
# - eval_transfer.py 里是通过 `model.readtext(...)` 来推理的，
#   MMOCR 引擎正好提供 readtext()，所以这里返回 MMOCR 实例即可。

TCM_ROOT = "/media/dongli911/Documents/Workflow/YanFangjun/UAW/TCM"
TCM_DET_CONFIG = "clip_db_r50_fpnc_prompt_gen_vis_1200e_ft_gen_ic15_adam_taiji_0120_184718/clip_db_r50_fpnc_prompt_gen_vis_1200e_ft_gen_ic15_adam_taiji.py"
TCM_DET_CKPT = "clip_db_r50_fpnc_prompt_gen_vis_1200e_ft_gen_ic15_adam_taiji_0120_184718/best_0_icdar2015_test_hmean-iou:hmean_epoch_45.pth"
sys.path.append(TCM_ROOT)  # 确保能 import 到 TCM 里的自定义模块

def _abspath_maybe(root: str, p: str) -> str:
    """把相对路径拼到 root 下；绝对路径保持不变。"""
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    if root is None:
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(root, p))


def build_tcm_mmocr(det_config_path: str = None,
                    det_ckpt_path: str = None,
                    device: str = "cuda:0"):
    """
    构造 TCM detector-only 的 MMOCR 引擎（用于推理/评测）。

    参数说明（与 eval_transfer.py 的调用保持兼容）：
    - det_config_path: TCM 的检测 config（.py）
    - det_ckpt_path: 你下载好的 detector 权重（.pth）
    - device: 推理设备，默认 cuda:0
    
    返回：
    - MMOCR 实例（具备 readtext()，可直接被 eval_transfer.py 使用）
    """
    
    # 延迟导入 MMOCR
    from mmocr.utils.ocr import MMOCR
    
    # 允许外部不传参时走默认常量
    det_config_path = det_config_path or TCM_DET_CONFIG
    det_ckpt_path = det_ckpt_path or TCM_DET_CKPT
    
    # 处理成绝对路径（兼容你用 AllConfig/... 这种相对写法）
    det_config_path = _abspath_maybe(TCM_ROOT, det_config_path)
    det_ckpt_path = _abspath_maybe(TCM_ROOT, det_ckpt_path)
    
    if not os.path.isfile(det_config_path):
        raise FileNotFoundError(f"TCM det_config not found: {det_config_path}")
    if not os.path.isfile(det_ckpt_path):
        raise FileNotFoundError(f"TCM det_ckpt not found: {det_ckpt_path}")
    
    cfg = Config.fromfile(det_config_path)
    
    # 1) import custom modules (关键：否则会出现 No module named 'ocrclip' / 自定义数据集、hook 找不到等问题)
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg["custom_imports"])
    
    # 2) 对齐 test.py：推理时关闭 cfg 里写死的 pretrained（避免额外加载或路径不一致导致报错）
    #    注意：即使你传了 rn50_path，这里也不会把它写进 cfg.model.pretrained，
    #    因为 test.py 明确将 pretrained 置为 None，依赖 det_ckpt 完整权重。
    if cfg.model.get("pretrained"):
        cfg.model.pretrained = None
    
    if cfg.model.get("neck"):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get("rfp_backbone") and neck_cfg.rfp_backbone.get("pretrained"):
                    neck_cfg.rfp_backbone.pretrained = None
        else:
            if cfg.model.neck.get("rfp_backbone") and cfg.model.neck.rfp_backbone.get("pretrained"):
                cfg.model.neck.rfp_backbone.pretrained = None
    
    # 3) 写临时 config 文件（MMOCR 需要文件路径而不是 Config 对象）
    tmp_dir = tempfile.mkdtemp(prefix="tcm_det_cfg_")
    tmp_cfg_path = os.path.join(tmp_dir, os.path.basename(det_config_path))
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg.pretty_text)
    
    # 4) detector-only MMOCR engine
    # 当同时提供 det_config 和 det_ckpt 时，MMOCR 会从配置文件推断模型
    # 关键：必须同时提供这两个参数，且不指定 det 参数
    model = MMOCR(
        det_config=tmp_cfg_path,
        det_ckpt=det_ckpt_path,
        recog=None,
        device=device
    )
    return model
