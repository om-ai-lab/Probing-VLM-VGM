# Probing-VLM-VGM

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2605.28132-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.28132)

Official code for **"Which Pretraining Paradigm Better Serves Spatial Intelligence? An Empirical Comparison of Vision-Language and Video Generation Models."**

This repository provides a unified frozen-feature probing framework for comparing **Vision-Language Models (VLMs)** and **Video Generation Models (VGMs)** across three representative axes of spatial intelligence:

- 🏷️ **Semantic tagging**: which object categories are visible in a video clip?
- 🧩 **Instance grouping**: which pixels across views belong to the same object instance?
- 🌐 **3D geometry prediction**: how well do frozen features support point maps, depth, and camera motion?

Our experiments show a clear complementarity: **VLMs are stronger at semantic and object-centric understanding**, while **VGMs provide more accessible dense geometry and camera-motion signals**. A simple feature-level fusion of VLM and VGM representations already improves both sides, suggesting a promising direction for stronger spatial-intelligence backbones.

## 📦 Repository Structure

```text
probing_vlm_vgm/        # Probe models, datasets, losses, metrics, and training entry point
configs/                # Hydra configs for semantic tagging, instance grouping, and 3D geometry
features/               # Frozen feature extraction wrappers for VLMs and VGMs
data/                   # User-provided datasets and extracted features (ignored by git)
ckpt/                   # User-provided model checkpoints (ignored by git)
```

## 🛠️ Installation

```bash
conda create -n probing-vlm-vgm python=3.11 cmake=3.14.0 -y
conda activate probing-vlm-vgm

# Install PyTorch. Please adjust the CUDA version to your system.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Install PyTorch3D. This can take a while.
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# Install dependencies and this package.
pip install -r requirements.txt
pip install -e .
```

The training code uses Hydra configs and expects `PROJECT_ROOT` to point to this repository:

```bash
export PROJECT_ROOT=/path/to/Probing-VLM-VGM
python -m probing_vlm_vgm.train --help
```


## 📚 Data Preparation

### ScanNet

ScanNet is distributed under its own Terms of Use and requires users to request access from the official ScanNet website. We do not provide ScanNet data or third-party mirrors. After obtaining access, prepare the processed ScanNet files and frozen features as:

```text
data/ScanNet/
  ScanNet-processed/
  FEAT/
```

ScanNet is used for:

- 🏷️ Semantic tagging
- 🧩 Instance grouping

### DL3DV

DL3DV is used for 3D geometry probing. Place processed geometry targets and frozen features as:

```text
data/DL3DV/
  DL3DV-processed/
  FEAT/
```

The geometry supervision follows the paper setup: VGGT-generated point maps, depth maps, camera poses, and confidence maps are used as probe targets.

## ❄️ Frozen Feature Extraction

The probe is trained on frozen intermediate features. We provide feature extraction wrappers under `features/`.

Supported model families include:

- 🎥 **VGMs**: WAN, CogVideoX, OpenSora, Aether
- 🖼️ **VLMs / visual models**: InternVL, Qwen2.5-VL, Qwen3-VL, VideoLLaMA3, DINOv2, V-JEPA, and related variants

Example commands:

```bash
# DL3DV VGM features, e.g. WAN
python -m features.run_dl3dv \
  --vfm wan \
  --subset all \
  --model-id ckpt/Wan2.1-T2V-14B-Diffusers \
  --prompt "" \
  --output-layers 20 \
  --t 749

# ScanNet VLM features, e.g. Qwen3-VL
python -m features.run_scannet \
  --vfm qwen3vl \
  --model-path ckpt/Qwen3-VL-8B-Instruct \
  --output-layers 22
```

Different feature extractors may require different checkpoint paths, input resolutions, or layer/timestep choices. See the docstring at the top of each `features/*/extract_features.py` file for model-specific examples.

> Note: Some backends, such as OpenSora and V-JEPA, require external code or checkpoints. We keep lightweight wrappers in this repository, but large vendored backends and checkpoints should be installed or downloaded separately.

## 🚀 Training and Evaluation

All tasks use the same entry point:

```bash
python -m probing_vlm_vgm.train experiment=<task>/<model> job_name=<run_name>
```

### Semantic Tagging

```bash
python -m probing_vlm_vgm.train \
  experiment=scannet_tagging/qwen3vl \
  job_name=qwen3vl
```

### Instance Grouping

```bash
python -m probing_vlm_vgm.train \
  experiment=scannet/wan-14b \
  job_name=wan14b
```

### 3D Geometry

```bash
python -m probing_vlm_vgm.train \
  experiment=dl3dv/wan-14b \
  job_name=wan14b
```

Hydra overrides can be used to change paths, feature layers, batch sizes, or probe settings:

```bash
python -m probing_vlm_vgm.train \
  experiment=dl3dv/qwen3vl \
  job_name=qwen3vl_layer22 \
  data.feat_root=/path/to/DL3DV/FEAT \
  feat_postfix=_layer22
```

## 🔗 Feature Fusion

We include configs for simple VLM+VGM feature-level fusion:

```bash
python -m probing_vlm_vgm.train \
  experiment=dl3dv/wan14b-qwen3vl-lnconcat \
  job_name=wan14b_qwen3vl_fusion
```

The fusion baseline normalizes frozen features from each model and concatenates them along the channel dimension before feeding them to the same probe.


```

## 🧾 Citation

If you find this project useful, please cite:

```bibtex
@misc{shen2026probingvlmvgm,
      title={Which Pretraining Paradigm Better Serves Spatial Intelligence? An Empirical Comparison of Vision-Language and Video Generation Models}, 
      author={Haozhan Shen and Tiancheng Zhao and Kangjia Zhao and Jianwei Yin},
      year={2026},
      eprint={2605.28132},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.28132}, 
}
```

## 🙏 Acknowledgments

This codebase builds on and adapts components from several excellent open-source projects, including **VidFM3D**, **VGGT**, **DUSt3R/Fast3R**, and feature extraction code or model interfaces from the evaluated VLM/VGM families. We thank the authors for making their implementations available.

Please refer to the original repositories and model cards for the licenses and terms of use of each dataset, model, and external dependency.
