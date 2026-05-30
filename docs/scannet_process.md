# ScanNet Preprocessing

This guide describes how to convert the official ScanNet data into the
`ScanNet-processed` layout used by this repository.

ScanNet is distributed under its own Terms of Use. Please request access from
the [official ScanNet repository](https://github.com/ScanNet/ScanNet) and do not
redistribute the dataset or unofficial mirrors.

## 1. Raw ScanNet Layout

The official download is scene-based. A raw scene usually contains files such as
`sceneXXXX_YY.sens`, `sceneXXXX_YY.aggregation.json`,
`sceneXXXX_YY_2d-instance-filt.zip`, and mesh/segmentation files.

Our preprocessing script does not read `.sens` directly. It expects each scene
to first be exported into RGB frames, camera poses, and intrinsics:

```text
data/ScanNet/scans/
  scans_train/
    scene0000_00/
      exported/
        color/0.jpg
        pose/0.txt
        intrinsic/intrinsic_color.txt
      scene0000_00_2d-instance-filt.zip
      scene0000_00.aggregation.json
  scans_val/
    sceneXXXX_YY/
      exported/
      sceneXXXX_YY_2d-instance-filt.zip
      sceneXXXX_YY.aggregation.json
  meta_data/
    scannetv2-labels.combined.tsv
```

## 2. Export `.sens` Files

If your downloaded ScanNet scenes are still in `.sens` form, first export each
scene with the official ScanNet `SensReader`:

```bash
python /path/to/ScanNet/SensReader/python/reader.py \
  --filename /path/to/scannet/scans/scene0000_00/scene0000_00.sens \
  --output_path /path/to/scannet/scans/scene0000_00/exported \
  --export_color_images \
  --export_poses \
  --export_intrinsics
```

Run this for all train and validation scenes you plan to use. The official
ScanNet scripts also provide download/export utilities; either route is fine as
long as each scene contains `exported/color`, `exported/pose`, and
`exported/intrinsic/intrinsic_color.txt`.

## 3. Organize Train and Validation Scenes

Our scripts expect `scans_train/` and `scans_val/`. If your official download is
a single `scans/sceneXXXX_YY/` directory, create a working layout by symlinking
scenes according to the official ScanNet v2 split files:

```bash
SCANNET_RAW=/path/to/scannet/scans
SCANNET_WORK=data/ScanNet/scans
SCANNET_META=/path/to/scannet/meta_data

mkdir -p ${SCANNET_WORK}/scans_train ${SCANNET_WORK}/scans_val
ln -sfn ${SCANNET_META} ${SCANNET_WORK}/meta_data

while IFS= read -r scene; do
  ln -sfn "${SCANNET_RAW}/${scene}" "${SCANNET_WORK}/scans_train/${scene}"
done < ${SCANNET_META}/scannetv2_train.txt

while IFS= read -r scene; do
  ln -sfn "${SCANNET_RAW}/${scene}" "${SCANNET_WORK}/scans_val/${scene}"
done < ${SCANNET_META}/scannetv2_val.txt
```

## 4. Build Processed Clips

This step samples each scene into an 81-frame clip, resizes RGB frames, copies
camera intrinsics, saves camera poses, and builds multi-view instance masks from
`sceneXXXX_YY_2d-instance-filt.zip`.

```bash
python -m probing_vlm_vgm.data.processing.scannet.process_scannet \
  --raw_root data/ScanNet/scans \
  --out_root data/ScanNet/ScanNet-processed \
  --split both
```

Then create `train.json` and `val.json`:

```bash
python -m probing_vlm_vgm.data.processing.scannet.create_split \
  --processed_root data/ScanNet/ScanNet-processed
```

## 5. Build Semantic-Tagging Labels

For the paper's ScanNet20 semantic tagging setup, build per-frame class pixel
counts and clip-level label diagnostics with:

```bash
python -m probing_vlm_vgm.data.processing.scannet.build_tag_labels \
  --raw_root data/ScanNet/scans \
  --processed_root data/ScanNet/ScanNet-processed \
  --label_map_tsv data/ScanNet/scans/meta_data/scannetv2-labels.combined.tsv \
  --num_classes 20 \
  --split both
```

If you also want ScanNet200 labels for additional experiments, rerun the command
with `--num_classes 200`.

## 6. Build CLIP Class Embeddings

The semantic tagging head can initialize class queries from CLIP text
embeddings. For ScanNet20, run:

```bash
python -m probing_vlm_vgm.data.processing.scannet.build_clip_class_embeds \
  --num_classes 20 \
  --out_path data/ScanNet/clip_class_embeds_20_vitl14.npy
```

For ScanNet200, use:

```bash
python -m probing_vlm_vgm.data.processing.scannet.build_clip_class_embeds \
  --num_classes 200 \
  --out_path data/ScanNet/clip_class_embeds_200_vitl14.npy
```

## 7. Expected Output

After preprocessing, the expected layout is:

```text
data/ScanNet/
  ScanNet-processed/
    train.json
    val.json
    class_names_20.json
    train_pos_rate_20.npy
    train/
      scene0000_00/
        frames/frame_00000.jpg ... frame_00080.jpg
        instance_masks.npy
        poses.npy
        intrinsic.txt
        metadata.sft
        tag_pixel_counts_20.npy
        tag_labels_20.npy
    val/
      sceneXXXX_YY/
        frames/
        instance_masks.npy
        poses.npy
        intrinsic.txt
        metadata.sft
        tag_pixel_counts_20.npy
        tag_labels_20.npy
  clip_class_embeds_20_vitl14.npy
```

Frozen features should be written separately under:

```text
data/ScanNet/FEAT/<model-name>/<split>/<scene_id>/feature_layer*.sft
```
