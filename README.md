# LookWhere (NeurIPS 2025 :tada:)
Official code for "LookWhere? Efficient Visual Recognition by Learning Where to Look and What to See from Self-Supervision"

arXiv link: https://arxiv.org/abs/2505.18051

LookWhere accelerates inference and fine-tuning by approximating full, deep representations with adaptive computation of predictions learned from distillation. It learns both where to look, with an efficient selector of locations, and what to see, with an expressive extractor of representations.

## Downstream
Setup
```bash
git clone
CD to
DL weights from HF
```


```python
import torch
from PIL import Image
from torchvision import transforms
from modeling import LookWhereDownstream

# user settings
high_res_img_size = 518  # size (height and width) of the high-res images
assert high_res_img_size % 14 == 0  # must be divisible by 14, since the patch size is 14x14
k_ratio = 0.1  # the fraction of high-res patches that are visible to the extractor
assert 0.01 < k_ratio < 0.9  # be safe
num_classes = 0  # number of classes, e.g., in ImageNet-1K num_classes=1_000
is_classification = True  # if this is False, then it will be in patch-segmentation mode
pretrained_params_path = "lookwhere_dinov2.pt"  # or "lookwhere_franca.pt", it _must_ be either!
device = "cuda"

# auto
num_high_res_patches = (high_res_img_size // 14)**2
k = int(k_ratio * num_high_res_patches)

# prepare input
transform = transforms.Compose([
    transforms.Resize(high_res_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
])
image = Image.open("ice_cream.jpg")
image = transform(image).unsqueeze(0).to(device)  # (bs, 3, high_res_img_size, high_res_img_size)

# create model and load pre-trained weights
lw = LookWhereDownstream(
    pretrained_params_path=pretrained_params_path,
    high_res_size=high_res_img_size,
    num_classes=num_classes,
    k=k,
    is_cls=is_classification,
    device=device
)

# for inference
with torch.no_grad():
    x_cls = lw(image)  # (bs, 768)
    # x_cls = lw(image)  # (bs, num_classes) if num_classes is not 0
    # x_patches = lw(image)  # (bs, 768, grid_size, grid_size) if num_classes is 0 and is_cls is False
    # x_patches = lw(image)  # (bs, num_classes, grid_size, grid_size) if num_classes is not 0 and is_cls is False
```

## Pre-training (JAX / TPU)
We ran all pre-training experiments on Google's TPUs. This code can be found in the directory: `original_jax_tpu_pretraining`

This pre-training pipeline is written in JAX and based off of the [deit3-jax codebase](https://github.com/affjljoo3581/deit3-jax). Please follow the setup instructions in their README to setup pre-training on TPUs for LookWhere. 

## Pre-training (PyTorch)
Sooner or later. Hopefully sooner.

# Please Cite
```bib
@misc{fuller2025lookwhereefficientvisualrecognition,
      title={LookWhere? Efficient Visual Recognition by Learning Where to Look and What to See from Self-Supervision}, 
      author={Anthony Fuller and Yousef Yassin and Junfeng Wen and Daniel G. Kyrollos and Tarek Ibrahim and James R. Green and Evan Shelhamer},
      year={2025},
      eprint={2505.18051},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.18051}, 
}
```
