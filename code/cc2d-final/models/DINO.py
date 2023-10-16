import torch
import os
import urllib

import mmcv
# from mmcv.runner import load_checkpoint
from utils import create_segmenter

def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"

def dinov2_vits14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, **kwargs)


def dinov2_vitb14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-B/14 model pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, **kwargs)


def dinov2_vitl14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, **kwargs)

def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    pretrained: bool = True,
    **kwargs,
):
    from dinov2.models import vision_transformer as vits

    model_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        path = os.path.join('/home1/qsyao/.cache/torch/hub/checkpoints/', f"{model_name}_pretrain.pth")
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model

if __name__ == "__main__":

    dinov2_vitb14_lc = dinov2_vitb14()

    dinov2_vitb14_lc = dinov2_vitb14_lc.cuda()
    dinov2_vitb14_lc.eval()

    test = torch.rand([1, 3, 518, 518]).cuda()
    out = dinov2_vitb14_lc(test)

    import ipdb; ipdb.set_trace()


    def load_config_from_url(url: str) -> str:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()


    HEAD_SCALE_COUNT = 3 # more scales: slower but better results, in (1,2,3,4,5)
    HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
    HEAD_TYPE = "ms" # in ("ms, "linear")

    backbone_name = "dinov2_vitb14"

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    if HEAD_TYPE == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
        print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

    import ipdb; ipdb.set_trace()

    model = create_segmenter(cfg, backbone_model=dinov2_vitb14_lc)
    # load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.cuda()
    model.eval()