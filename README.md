# LookWhere (NeurIPS 2025 :tada:)
Official code for "LookWhere? Efficient Visual Recognition by Learning Where to Look and What to See from Self-Supervision"

arXiv link: https://arxiv.org/abs/2505.18051

LookWhere accelerates inference and fine-tuning by approximating full, deep representations with adaptive computation of predictions learned from distillation. It learns both where to look, with an efficient selector of locations, and what to see, with an expressive extractor of representations.

## Downstream
### Setup environment
Clone the repo, download weights, and install einops and timm if not already installed. If you are only using LookWhere-DINOv2 (which is better than our LookWhere-Franca), then you only need to download it.
```bash
git clone https://github.com/antofuller/lookwhere.git
cd lookwhere
wget https://huggingface.co/antofuller/lookwhere/resolve/main/lookwhere_dinov2.pt
wget https://huggingface.co/antofuller/lookwhere/resolve/main/lookwhere_franca.pt
pip install einops
pip install timm
```

### Setup model for fine-tuning or inference
We've made it as simple as possible to fine-tune or directly use LookWhere. Here are the user settings:
* high_res_img_size: The size (height and width) of the high-res images. LookWhere is pre-trained at 518x518, thus if you choose another value for this you should fine-tune it to adjust to the different resolution. We've fine-tuned up to 1036x1036, but think you can go higher without issue. Because our patch size is 14x14, this value needs to be evenly divisible by 14.
* k_ratio: The fraction of high-res patches that are visible to the extractor. This should be between 0 and 1, lower values make LookWhere faster but likely reduce accuracy (depending on the task of course). You should try different values of k to choose an acceptable speed-accuracy operating point.
* num_classes: The number of classes per image (if in classification mode) or per patch (if in segmentation mode). This just attaches a linear head on top of our pre-trained LookWhere. If this is 0, then no head will be used and LookWhere will return extracted features.
* is_classification: The task type, True puts LookWhere in classification mode, False in segmentation mode. If num_classes is 0, then it returns the CLS token (in classification mode) or all (interpolated high-res) patch tokens (in segmentation mode). If num_classes is _not_ 0, then it returns the logits per image (in classification mode) or logits per patch (in segmentation mode).
* pretrained_params_path: This needs to be either "lookwhere_dinov2.pt" or "lookwhere_franca.pt". A directory can precede the file name and it should work without issue.
```python
import torch
from PIL import Image
from torchvision import transforms
from modeling import LookWhereDownstream

# user settings
high_res_img_size = 518
assert high_res_img_size % 14 == 0
k_ratio = 0.1
assert 0.01 < k_ratio < 0.99
num_classes = 0
is_classification = True
pretrained_params_path = "lookwhere_dinov2.pt"
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
image = Image.open("ice_cream.jpg")  # shout-out to lucidrains and ice cream: https://github.com/lucidrains
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

If you only need the selector map
```python
with torch.no_grad():
    selector_map = lw.selector(image)["selector_map"]  # (bs, num_high_res_patches)
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
