import os
import json
import random
import numpy as np
import torch
import tqdm
import cv2
import skimage.io as sio
import argparse
from datetime import datetime
from PIL import Image
import glob
from pycocotools import mask as mask_utils

# MMDetection imports - updated for MMDetection 3.x
from mmengine.config import Config
from mmengine.runner import Runner

# from mmengine.logging import MMLogger # Runner handles logging setup

from mmdet.apis import init_detector

# TRANSFORMS registry might be needed if defining custom transforms
# from mmdet.registry import TRANSFORMS
# BoxType not typically needed directly
# from mmdet.structures.bbox import BoxType
from mmdet.utils import register_all_modules
from visualizations import (
    visualize_dataset_examples,
    plot_class_distribution,
    plot_training_performance,
    visualize_predictions,
    analyze_model_predictions
)

# ------------------------------------------------------------------------------
# Utility Functions (Unchanged)
# ------------------------------------------------------------------------------


def setup_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Check if CUDA is available before setting seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Add determinism flag if needed (can slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure dir exists
    with open(file_path, "w") as f:
        # Use numpy encoder for compatibility if needed
        json.dump(data, f, indent=4, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def create_output_dir(root_dir):
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(root_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def encode_rle_mask(binary_mask):
    """Encode binary mask to RLE format."""
    # Ensure input is contiguous C-ordered numpy array of type uint8
    mask_array = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(mask_array)
    # Decode bytes to string for JSON serialization
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_rle_mask(rle_mask):
    """Decode RLE format to binary mask."""
    # Encode string back to bytes if needed (depends on how RLE was stored)
    if isinstance(rle_mask["counts"], str):
        rle_mask["counts"] = rle_mask["counts"].encode("utf-8")
    return mask_utils.decode(rle_mask)


def read_image(file_path):
    """Read image file and return as numpy array."""
    # Use mmcv's imread for consistency
    # return mmcv.imread(file_path)
    # Or stick to PIL/OpenCV if preferred
    img = np.array(Image.open(file_path))
    # Ensure 3 channels if needed by the model (e.g., for grayscale)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # Handle RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def read_mask(file_path):
    """Read mask file and return as numpy array."""
    # Use skimage.io as before
    return sio.imread(file_path)


# ------------------------------------------------------------------------------
# Dataset Preparation (Mostly Unchanged - ensure COCO format is correct)
# ------------------------------------------------------------------------------


def extract_masks(mask_path):
    """Extract individual masks from a mask file."""
    mask = read_mask(mask_path)
    if mask is None or mask.size == 0:
        print(f"Warning: Could not read or empty mask file: {mask_path}")
        return []
    unique_values = np.unique(mask)
    unique_values = unique_values[unique_values > 0]  # Remove background

    instance_masks = []
    for value in unique_values:
        # Already boolean, convert to uint8 later if needed
        binary_mask = mask == value
        # Check if mask is not empty
        if np.any(binary_mask):
            instance_masks.append(binary_mask)  # Keep as boolean for now

    return instance_masks


def get_bbox_from_mask(binary_mask):
    """Get bounding box coordinates [x, y, w, h] from a binary mask."""
    if not np.any(binary_mask):  # Handle empty mask
        return [0, 0, 0, 0]
    # Convert boolean mask to uint8 for findContours
    binary_mask_uint8 = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return [0, 0, 0, 0]

    x, y, w, h = cv2.boundingRect(contours[0])
    return [float(x), float(y), float(w), float(h)]  # Ensure float for COCO


def prepare_coco_annotations(data_dir, output_dir):
    """Create COCO format annotations from the dataset."""
    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return None
    sample_folders = [
        d
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]

    # Define class names and mapping
    # 0-indexed for model, 1-indexed for COCO category_id
    class_names = ["class1", "class2", "class3", "class4"]
    coco_categories = [
        {"id": i + 1, "name": name} for i, name in enumerate(class_names)
    ]

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": coco_categories,
    }

    annotation_id = 0
    image_id_counter = 0  # Use a simple counter for image IDs

    print("Preparing dataset annotations...")
    for sample_name in tqdm.tqdm(sample_folders):
        sample_dir = os.path.join(train_dir, sample_name)
        # Use original name structure
        image_file = os.path.join(sample_dir, "image.tif")

        if not os.path.exists(image_file):
            continue

        # Read image to get dimensions using a robust method
        try:
            # img = cv2.imread(image_file) # Can fail on some TIFs
            img = read_image(image_file)
            if img is None:
                print(f"Warning: Failed to read image {image_file}, skipping.")
                continue
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Error reading image {image_file}: {e}, skipping.")
            continue

        # Relative path for file_name in COCO
        relative_image_path = os.path.relpath(image_file, data_dir)

        current_image_id = image_id_counter
        coco_data["images"].append(
            {
                "id": current_image_id,
                # Ensure forward slashes
                "file_name": relative_image_path.replace("\\", "/"),
                "height": height,
                "width": width,
            }
        )
        image_id_counter += 1

        for class_idx, class_type in enumerate(class_names):
            mask_file = os.path.join(
                sample_dir, f"{class_type}.tif"
            )  # Original name structure
            if not os.path.exists(mask_file):
                continue

            # Extract instance masks from class mask
            try:
                instance_masks = extract_masks(mask_file)
            except Exception as e:
                print(
                    f"Error extracting masks from {mask_file}: {e}"
                )
                continue

            # Create annotations for each instance
            for (
                instance_mask
            ) in instance_masks:  # instance_mask is boolean here
                # Get bounding box
                bbox = get_bbox_from_mask(instance_mask)
                if bbox[2] <= 0 or bbox[3] <= 0:  # Skip empty boxes
                    continue

                # Calculate area
                area = int(np.sum(instance_mask))
                if area <= 0:  # Skip empty masks
                    continue

                rle_mask = encode_rle_mask(instance_mask)

                # Add annotation
                coco_data["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "category_id": class_idx
                        + 1,  # COCO category_id is 1-based
                        "bbox": bbox,
                        "area": area,
                        "segmentation": rle_mask,
                        "iscrowd": 0,  # Assuming no crowd regions
                    }
                )
                annotation_id += 1

    # Save the combined annotations
    combined_ann_file = os.path.join(output_dir, "annotations_combined.json")
    save_json(coco_data, combined_ann_file)
    print(f"Saved combined annotations to {combined_ann_file}")

    return coco_data


def split_train_val(coco_data, data_dir, output_dir, val_ratio=0.2):
    """Split the dataset into training and validation sets."""
    if not coco_data or not coco_data["images"]:
        print("Error: No image data found to split.")
        return None, None

    num_images = len(coco_data["images"])
    image_ids = [img["id"] for img in coco_data["images"]]
    random.shuffle(image_ids)

    split_point = int(num_images * (1 - val_ratio))
    train_image_ids = set(image_ids[:split_point])
    val_image_ids = set(image_ids[split_point:])

    # Create training dataset
    train_data = {
        "images": [
            img for img in coco_data["images"] if img["id"] in train_image_ids
        ],
        "categories": coco_data["categories"],
        "annotations": [
            anno
            for anno in coco_data["annotations"]
            if anno["image_id"] in train_image_ids
        ],
    }

    # Create validation dataset
    val_data = {
        "images": [
            img for img in coco_data["images"] if img["id"] in val_image_ids
        ],
        "categories": coco_data["categories"],
        "annotations": [
            anno
            for anno in coco_data["annotations"]
            if anno["image_id"] in val_image_ids
        ],
    }

    # Save split annotations
    train_ann_file = os.path.join(output_dir, "train_annotations.json")
    val_ann_file = os.path.join(output_dir, "val_annotations.json")
    save_json(train_data, train_ann_file)
    save_json(val_data, val_ann_file)
    print(f"Saved train annotations to {train_ann_file}")
    print(f"Saved val annotations to {val_ann_file}")

    return train_ann_file, val_ann_file


# ------------------------------------------------------------------------------
# Model Configuration (MMDetection 3.x Style)
# ------------------------------------------------------------------------------


def get_model_config(args, train_ann_file, val_ann_file, test_ann_file):
    """Create MMDetection 3.x configuration for Mask R-CNN model."""
    # Define class names and metainfo (consistent with dataset prep)
    class_names = ("class1", "class2", "class3", "class4")
    metainfo = dict(classes=class_names)
    num_classes = len(class_names)  # Should be 4

    # Base configuration (can inherit from a file like _base_/...)
    # Or define everything manually
    cfg = Config()

    cfg.default_scope = "mmdet"

    # ----- Runtime Settings -----
    cfg.work_dir = os.path.join(args.output_dir, "work_dir")
    cfg.load_from = None  # Or path to pretrained weights
    cfg.resume = False  # Or auto resume with 'auto'

    cfg.env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
        dist_cfg=dict(backend="nccl"),
    )

    # Set deterministic=True for reproducibility (slower)
    cfg.randomness = dict(seed=args.seed, deterministic=False)

    # ----- Visualizer Settings -----
    cfg.vis_backends = [dict(type="LocalVisBackend")]
    cfg.visualizer = None

    # ----- Logger Settings -----
    cfg.log_level = "INFO"
    cfg.log_processor = dict(
        type="LogProcessor", window_size=50, by_epoch=True
    )

    # ----- Model Definition -----
    cfg.model = dict(
        type="MaskRCNN",
        data_preprocessor=dict(
            type="DetDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
        ),
        backbone=dict(
            type="ResNet",
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type="BN", requires_grad=True),
            norm_eval=True,
            style="pytorch",
            init_cfg=dict(
                type="Pretrained", checkpoint="torchvision://resnet18"
            ),
        ),
        neck=dict(
            type="FPN",
            in_channels=[64, 128, 256, 512],
            out_channels=128,
            num_outs=5,
        ),
        rpn_head=dict(
            type="RPNHead",
            in_channels=128,  # Match FPN out_channels
            feat_channels=128,  # Match FPN out_channels
            anchor_generator=dict(
                type="AnchorGenerator",
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
            ),
            loss_cls=dict(
                type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
        roi_head=dict(
            type="StandardRoIHead",
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(
                    type="RoIAlign", output_size=7, sampling_ratio=0
                ),
                out_channels=128,  # Match FPN out_channels
                featmap_strides=[4, 8, 16, 32],
            ),
            bbox_head=dict(
                type="Shared2FCBBoxHead",
                in_channels=128,  # Match FPN out_channels
                fc_out_channels=512,  # Reduced from 1024
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
            mask_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(
                    type="RoIAlign", output_size=14, sampling_ratio=0
                ),
                out_channels=128,  # Match FPN out_channels
                featmap_strides=[4, 8, 16, 32],
            ),
            mask_head=dict(
                type="FCNMaskHead",
                num_convs=2,  # Reduced from 4
                in_channels=128,  # Match FPN out_channels
                conv_out_channels=128,  # Reduced from 256
                num_classes=num_classes,
                loss_mask=dict(
                    type="CrossEntropyLoss", use_mask=True, loss_weight=1.0
                ),
            ),
        ),
        # model training and testing settings
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,  # For maskrcnn this is True
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type="nms", iou_threshold=0.5),
                max_per_img=500,  # Adjust based on expected nuclei count
                mask_thr_binary=0.5,
            ),
        ),
    )

    # ----- Dataset Settings -----
    dataset_type = "CocoDataset"
    data_root = args.data_dir  # Set data root

    train_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
        dict(type="RandomFlip", prob=0.5),
        dict(type="Pad", size_divisor=32),
        dict(type="PackDetInputs"),
    ]

    test_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
        # Don't load annotations during testing usually
        # dict(type='LoadAnnotations', with_bbox=True, with_mask=True), # Not
        # needed for inference usually
        dict(type="Pad", size_divisor=32),
        # PackDetInputs automatically handles keys unless specified otherwise
        dict(
            type="PackDetInputs",
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
            ),
        ),
    ]

    train_ann_file_abs = os.path.abspath(train_ann_file)
    val_ann_file_abs = os.path.abspath(val_ann_file)
    test_ann_file_abs = os.path.abspath(test_ann_file)

    # Dataloaders
    cfg.train_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        # InfiniteSampler is default for epoch-based
        sampler=dict(type="DefaultSampler", shuffle=True),
        batch_sampler=dict(type="AspectRatioBatchSampler"),
        # Groups images by aspect ratio
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,  # Add metainfo
            ann_file=train_ann_file_abs,
            data_prefix=dict(img=""),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            # Filter small/empty GT
            pipeline=train_pipeline,
            backend_args=None,
        ),
    )

    cfg.val_dataloader = dict(
        batch_size=1,  # Validation typically uses batch size 1
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file=val_ann_file_abs,
            data_prefix=dict(img=""),
            test_mode=True,
            pipeline=test_pipeline,
            backend_args=None,
        ),
    )

    # Test dataloader (for prediction/submission generation)
    # If test set has GT for evaluation, use val_dataloader structure
    # If test set has no GT (like a competition), use structure below
    cfg.test_dataloader = dict(
        batch_size=1,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,  # Add metainfo
            ann_file=test_ann_file_abs,  # Annotation file with image info ONLY
            data_prefix=dict(img=""),
            test_mode=True,
            pipeline=test_pipeline,
            backend_args=None,
        ),
    )

    # ----- Evaluator Settings -----
    # Using COCO metrics for bbox and segmentation
    cfg.val_evaluator = dict(
        type="CocoMetric",
        ann_file=val_ann_file_abs,  # Path to val ground truth
        metric=["bbox", "segm"],
        format_only=False,
        backend_args=None,
    )
    # Define test evaluator similarly if GT available, or set to
    # None/format_only
    cfg.test_evaluator = dict(
        type="CocoMetric",
        ann_file=test_ann_file_abs,
        # Path to test GT (if available) or image info file
        metric=["bbox", "segm"],
        format_only=True,  # Set True if only generating prediction file
        outfile_prefix=os.path.join(cfg.work_dir, "test_submission"),
        # Prefix for output JSON
        backend_args=None,
    )

    # ----- Training Schedule -----
    # Training loop configuration
    cfg.train_cfg = dict(
        type="EpochBasedTrainLoop",
        max_epochs=args.epochs,
        val_interval=args.checkpoint_interval,
    )  # Use checkpoint interval for validation freq
    cfg.val_cfg = dict(type="ValLoop")  # Default validation loop
    cfg.test_cfg = dict(type="TestLoop")  # Default test loop

    # Optimizer Wrapper
    cfg.optim_wrapper = dict(
        type="OptimWrapper",
        optimizer=dict(
            type="SGD",
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=0.0001,
        ),
        # Add gradient clipping if needed
        # clip_grad=dict(max_norm=35, norm_type=2)
    )

    # Learning Rate Scheduler
    cfg.param_scheduler = [
        dict(
            type="LinearLR",  # Warmup
            start_factor=0.001,
            by_epoch=False,  # Warmup per iteration
            begin=0,
            end=500,
        ),  # Number of warmup iterations
        dict(
            type="MultiStepLR",
            by_epoch=True,  # Step per epoch
            begin=0,
            end=args.epochs,
            milestones=args.decay_steps,  # Epochs to decay LR
            gamma=0.1,
        ),  # Decay factor
    ]

    # ----- Hooks -----
    # Default hooks from MMDetection baseline configs
    cfg.default_hooks = dict(
        timer=dict(type="IterTimerHook"),
        logger=dict(type="LoggerHook", interval=50),  # Log every 50 iterations
        param_scheduler=dict(type="ParamSchedulerHook"),
        checkpoint=dict(
            type="CheckpointHook",
            # Save checkpoint interval (epochs)
            interval=args.checkpoint_interval,
            save_best="auto",  # Automatically save best based on val metric
            max_keep_ckpts=3,
        ),  # Keep last 3 checkpoints
        sampler_seed=dict(type="DistSamplerSeedHook"),
        # Visualization hook - disabled by default, enable for debugging
        visualization=dict(
            type="DetVisualizationHook", draw=False, interval=1
        ),
    )  # Draw predictions every 'interval' epochs during validation

    return cfg


# ------------------------------------------------------------------------------
# Test Dataset Preparation (Updated)
# ------------------------------------------------------------------------------


def prepare_test_dataset(data_dir, output_dir):
    """Prepare test dataset image info file for inference (COCO format)
    USING the provided filename-to-ID mapping (handles list or dict format)."""
    test_image_dir = os.path.join(data_dir, "test_release")
    mapping_file = os.path.join(data_dir, "test_image_name_to_ids.json")

    if not os.path.exists(test_image_dir):
        print(f"Error: Test image directory not found: {test_image_dir}")
        return None
    if not os.path.exists(mapping_file):
        return None

    test_files = glob.glob(os.path.join(test_image_dir, "*.tif"))

    # Load the mapping file
    try:
        with open(mapping_file, "r") as f:
            raw_mapping_data = json.load(f)
        print(f"Loaded mapping data from: {mapping_file}")
    except Exception as e:
        print(f"Error loading or parsing mapping file {mapping_file}: {e}")
        return None

    image_name_to_id_map = {}
    if isinstance(raw_mapping_data, list):
        print("Mapping data is a list. Converting to dictionary...")
        try:
            # --- IMPORTANT: Verify the key names ('file_name', 'id') below ---
            # --- match the actual keys in your JSON file ---
            expected_name_key = (
                "file_name"  # Or 'name', 'filename', etc. CHECK YOUR JSON!
            )
            expected_id_key = "id"  # Or 'image_id', etc. CHECK YOUR JSON!

            for item in raw_mapping_data:
                if not isinstance(item, dict):
                    continue
                if (
                    expected_name_key not in item
                    or expected_id_key not in item
                ):
                    continue
                image_name = item[expected_name_key]
                image_id = item[expected_id_key]
                image_name_to_id_map[image_name] = image_id
        except Exception as e:
            print(f"Error converting list mapping data to dictionary: {e}")
            return None
    elif isinstance(raw_mapping_data, dict):
        print("Mapping data is already a dictionary.")
        image_name_to_id_map = raw_mapping_data  # Use it directly
    else:
        return None

    if not image_name_to_id_map:
        print(
            "Error: The image name to ID map is empty"
        )
        return None

    # Define categories (consistent with training)
    # Make sure these match your actual classes
    class_names = ["class1", "class2", "class3", "class4"]
    coco_categories = [
        {"id": i + 1, "name": name} for i, name in enumerate(class_names)
    ]

    # Create test image info file (COCO format without annotations)
    test_data = {
        "images": [],
        "annotations": [],
        "categories": coco_categories,
    }

    print(
        f"Processing {len(test_files)} test images"
    )
    processed_count = 0
    skipped_count = 0
    for test_file in test_files:
        image_name = os.path.basename(test_file)

        # Now use .get() on the guaranteed dictionary
        correct_image_id = image_name_to_id_map.get(image_name)

        if correct_image_id is None:
            skipped_count += 1
            continue

        # Ensure the ID is an integer
        try:
            correct_image_id = int(correct_image_id)
        except ValueError:
            print(
                f"Warning: Invalid ID '{correct_image_id}'"
            )
            skipped_count += 1
            continue

        # Read image dimensions
        try:
            # Assuming read_image function exists and works
            img = read_image(test_file)
            if img is None:
                print(
                    f"Warning: Failed to read {test_file}"
                )
                skipped_count += 1
                continue
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Error reading test image {test_file}: {e}, skipping.")
            skipped_count += 1
            continue

        relative_image_path = os.path.relpath(test_file, data_dir)

        # Add image info using the CORRECT ID from the mapping
        test_data["images"].append(
            {
                "id": correct_image_id,
                "file_name": relative_image_path.replace("\\", "/"),
                "height": height,
                "width": width,
            }
        )
        processed_count += 1

    if skipped_count > 0:
        print(
            f"Warning: Skipped {skipped_count} test images"
        )
    if processed_count == 0 and len(test_files) > 0:
        print(
            "Error: No test images were successfully processed."
        )
        return None

    # Sort images by ID for consistency (optional but good practice)
    test_data["images"].sort(key=lambda x: x["id"])

    # Save the test image info file (which now has the correct IDs)
    test_ann_file = os.path.join(output_dir, "test_annotations.json")
    # Assuming save_json function exists and works
    save_json(test_data, test_ann_file)
    print(f"Saved test image info file (with correct IDs) to {test_ann_file}")

    return test_ann_file


# ------------------------------------------------------------------------------
# Inference and Submission (Updated for MMDetection 3.x)
# ------------------------------------------------------------------------------


def predict_masks_and_create_submission(model, cfg, output_dir):
    """Run inference using Runner and format results for submission."""

    print("Running inference using Runner...")
    runner = Runner.from_cfg(cfg)  # Rebuild runner with the final config

    runner.model = model
    metrics = runner.test()

    print("Inference finished.")
    # Will be empty if format_only=True
    print("Metrics (if evaluated):", metrics)

    outfile_prefix = cfg.test_evaluator["outfile_prefix"]
    expected_submission_file = f"{outfile_prefix}.segm.json"

    if os.path.exists(expected_submission_file):
        print(
            f"Submission JSON at{expected_submission_file}"
        )
        # Optional: Rename or move the file if needed
        final_submission_path = os.path.join(output_dir, "submission.json")
        os.rename(expected_submission_file, final_submission_path)
        print(f"Renamed submission file to: {final_submission_path}")
        return final_submission_path
    else:
        print(
            f"Error: file not at {expected_submission_file}"
        )
        print("Please check the test evaluator configuration and work_dir.")
        # Fallback: Try manual inference if runner fails (less ideal)
        # return run_manual_inference(model, cfg, output_dir) # Implement if
        # needed
        return None


# Optional: Fallback manual inference loop (less recommended than Runner)
def run_manual_inference(model, cfg, output_dir):
    print("Running manual inference (fallback)...")
    test_dataloader_cfg = cfg.test_dataloader
    # Build the dataset and dataloader manually
    from mmengine.dataset import build_dataset, pseudo_collate

    # Re-use parts of val loop logic maybe? No, better to loop manually.
    from mmdet.datasets import build_dataloader

    test_dataset = build_dataset(test_dataloader_cfg["dataset"])
    test_loader_cfg = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        collate_fn=pseudo_collate,  # Use pseudo_collate for list output
    )
    test_loader = build_dataloader(test_dataset, test_loader_cfg)

    results_list = []
    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            # Data is now a list of dicts from PackDetInputs
            # Need to potentially move data to the correct device
            # inference_detector handles this internally usually

            # Get img_id from PackDetInputs meta_keys
            img_id = data[0]["data_samples"].img_id

            # Run inference on the single image data dict
            result = model.test_step(data)[0]  # Use test_step for single batch

            # Process results (similar to original predict_masks but using
            # DetDataSample)
            pred_instances = result.pred_instances  # Access predictions here

            # Check if masks are present
            if "masks" not in pred_instances:
                continue

            # Move results to CPU and numpy
            pred_instances = pred_instances.to("cpu")
            scores = pred_instances.scores.numpy()
            bboxes = pred_instances.bboxes.numpy()  # Format [x1, y1, x2, y2]
            masks = pred_instances.masks.numpy()  # Boolean masks [N, H, W]

            for i in range(len(scores)):
                score = scores[i]
                if score < 0.05:  # Confidence threshold
                    continue

                bbox = bboxes[i]
                mask = masks[i]  # This is a binary mask [H, W]

                # Convert to RLE format
                mask_rle = encode_rle_mask(mask)

                # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height]
                x1, y1, x2, y2 = bbox
                bbox_xywh = [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                ]

                # Append result in COCO format
                results_list.append(
                    {
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": bbox_xywh,
                        "score": float(score),
                        "segmentation": mask_rle,
                    }
                )

    # Save results
    submission_file = os.path.join(output_dir, "submission_manual.json")
    save_json(results_list, submission_file)
    print(f"Saved manual submission to {submission_file}")
    return submission_file


# ------------------------------------------------------------------------------
# Main Function (Updated for MMDetection 3.x Runner)
# ------------------------------------------------------------------------------


def main():
    """Main function to run the full pipeline."""
    # Register all modules from mmdet - Use default scope initialization
    # Usually safer for finding modules
    register_all_modules(init_default_scope=True)

    parser = argparse.ArgumentParser(
        description="Nuclei Instance Segmentation with MMDetection 3.x"
    )

    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="hw3-data-release",
        help="Path to dataset directory",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training (per GPU)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
        help="Base learning rate (adjust based on total batch size)",
    )
    parser.add_argument(
        "--decay_steps",
        type=int,
        nargs="+",
        default=[20, 40],
        help="Epochs at which to decay learning rate",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5,
        help="Interval (epochs) for saving checkpoints and running validation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Parent directory to save outputs",
    )

    # Action flags
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and go directly to inference",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to specific checkpoint for inference (overrides latest)",
    )

    # Visualization flags
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )
    parser.add_argument(
        "--vis_samples",
        type=int,
        default=3,
        help="Number of samples to visualize",
    )

    args = parser.parse_args()

    # Set random seed
    setup_seed(args.seed)

    # Create timestamped output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")

    # --- 1. Prepare Dataset Annotations ---
    print("Preparing dataset annotations...")
    # Save annotations directly in output_dir - no separate annotations
    # subfolder

    coco_data = prepare_coco_annotations(args.data_dir, output_dir)
    if coco_data is None:
        print("Failed to prepare COCO data. Exiting.")
        return

    train_ann_file, val_ann_file = split_train_val(
        coco_data, args.data_dir, output_dir
    )
    if train_ann_file is None or val_ann_file is None:
        print("Failed to split train/val data. Exiting.")
        return

    # Prepare test dataset image info file (needed for config and inference)
    test_ann_file = prepare_test_dataset(args.data_dir, output_dir)
    if test_ann_file is None:
        print("Failed to prepare test data file. Exiting.")
        return

    # --- 1.1 Dataset Visualization (if requested) ---
    if args.visualize:
        print("Generating dataset visualizations...")
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Visualize dataset examples
        visualize_dataset_examples(
            args.data_dir, vis_dir, max_samples=args.vis_samples
        )

        # Plot class distribution
        dataset_stats = plot_class_distribution(args.data_dir, vis_dir)
        print("Dataset statistics:")
        print(f"  Total images: {dataset_stats['image_count']}")
        print(f"  Average image size: {dataset_stats['avg_image_size']}")
        print(f"  Class counts: {dataset_stats['class_counts']}")

    # --- 2. Get Model Configuration ---
    # Pass annotation file paths to config function
    cfg = get_model_config(args, train_ann_file, val_ann_file, test_ann_file)
    # Update work_dir in the config (already done inside get_model_config)
    print(f"Configuration prepared. Work directory set to: {cfg.work_dir}")

    # --- 3. Training ---
    runner = None  # Initialize runner variable
    if not args.skip_train:
        print("Initializing training runner...")
        # Create a runner from the config
        runner = Runner.from_cfg(cfg)

        print("Starting model training...")
        runner.train()
        print("Training finished.")
    else:
        print("Skipping training as requested.")

    # --- 3.1 Visualize Training Performance (if requested) ---
    if args.visualize and not args.skip_train:
        print("Generating training performance visualizations...")
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Look for the training log file
        log_file = os.path.join(cfg.work_dir, "20250429_000000.log.json")
        if os.path.exists(log_file):
            plot_training_performance(log_file, vis_dir)
        else:
            # Try to find any log file
            log_files = glob.glob(os.path.join(cfg.work_dir, "*.log.json"))
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                plot_training_performance(latest_log, vis_dir)
            else:
                print(
                    f"Warning: No log files found in {cfg.work_dir}"
                )

    # --- 4. Inference ---
    print("Preparing for inference...")

    # Determine checkpoint path for inference
    if args.checkpoint_path:
        checkpoint_file = args.checkpoint_path
        if not os.path.exists(checkpoint_file):
            print(f"Error: Specified checkpoint not found: {checkpoint_file}")
            return
    elif (
        not args.skip_train and runner
    ):  # Use last checkpoint from training run
        checkpoint_file = os.path.join(cfg.work_dir, "latest.pth")
        if not os.path.exists(checkpoint_file):
            print(
                f"Warning: latest.pth not found in {cfg.work_dir}"
            )
            # Try finding any .pth file as a fallback
            checkpoints = glob.glob(os.path.join(cfg.work_dir, "*.pth"))
            if checkpoints:
                checkpoint_file = max(
                    checkpoints, key=os.path.getctime
                )  # Get newest
                print(f"Using newest checkpoint found: {checkpoint_file}")
            else:
                print("Error: No checkpoint found after training.")
                return
    else:  # Skipping training and no specific checkpoint provided
        # Attempt to find latest checkpoint in the config's work_dir
        checkpoint_file = os.path.join(cfg.work_dir, "latest.pth")
        if not os.path.exists(checkpoint_file):
            checkpoints = glob.glob(os.path.join(cfg.work_dir, "*.pth"))
            if checkpoints:
                checkpoint_file = max(
                    checkpoints, key=os.path.getctime
                )  # Get newest
                print(f"Found existing newest checkpoint: {checkpoint_file}")
            else:
                return

    print(f"Using checkpoint for inference: {checkpoint_file}")

    # Update config for inference (load the specific checkpoint)
    cfg.load_from = checkpoint_file

    # Initialize detector model directly
    print("Initializing model for inference...")
    model = init_detector(
        cfg,
        checkpoint_file,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # --- 4.1 Visualize Predictions (if requested) ---
    if args.visualize and model is not None:
        print("Generating model prediction visualizations...")
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Visualize predictions on test images
        visualize_predictions(
            model, args.data_dir, vis_dir, num_samples=args.vis_samples
        )

        # Analyze model predictions
        analyze_model_predictions(model, args.data_dir, vis_dir)

    # Run inference and create submission file using the Runner's test loop
    print("Running inference using Runner...")
    runner = Runner.from_cfg(cfg)
    runner.model = model
    metrics = runner.test()

    print("Inference finished.")
    print("Metrics (if evaluated):", metrics)

    # Find the generated results file
    outfile_prefix = cfg.test_evaluator["outfile_prefix"]
    expected_submission_file = f"{outfile_prefix}.segm.json"

    if os.path.exists(expected_submission_file):

        # *** FIX: Fix the submission file to ensure correct image IDs ***
        try:
            # Load test annotations to get the correct image ID mapping
            with open(test_ann_file, "r") as f:
                test_annotations = json.load(f)

            # Load the mapping file
            mapping_file = os.path.join(
                args.data_dir, "test_image_name_to_ids.json"
            )
            with open(mapping_file, "r") as f:
                mapping_data = json.load(f)

            # Convert list format to dictionary if needed
            image_name_to_id_map = {}
            if isinstance(mapping_data, list):
                for item in mapping_data:
                    if (
                        isinstance(item, dict)
                        and "file_name" in item
                        and "id" in item
                    ):
                        image_name_to_id_map[item["file_name"]] = item["id"]
            else:
                image_name_to_id_map = mapping_data

            # Load the generated submission file
            with open(expected_submission_file, "r") as f:
                submission_data = json.load(f)

            filename_to_correct_id = {
                os.path.basename(img["file_name"]): img["id"]
                for img in test_annotations["images"]
            }

            # Fix IDs in submission data
            test_dataset = runner.test_dataloader.dataset
            fixed_submission = []
            fixed_count = 0

            # Get a dictionary that maps dataset index to image filename
            idx_to_filename = {}
            for idx in range(len(test_dataset)):
                data_info = test_dataset.get_data_info(idx)
                if "img_path" in data_info:
                    idx_to_filename[idx] = os.path.basename(
                        data_info["img_path"]
                    )
                elif "file_name" in data_info:
                    idx_to_filename[idx] = os.path.basename(
                        data_info["file_name"]
                    )

            # Create a mapping from dataset internal IDs to correct IDs
            internal_id_to_correct_id = {}
            for idx, filename in idx_to_filename.items():
                if filename in filename_to_correct_id:
                    internal_id_to_correct_id[idx] = filename_to_correct_id[
                        filename
                    ]
                elif filename in image_name_to_id_map:
                    internal_id_to_correct_id[idx] = image_name_to_id_map[
                        filename
                    ]

            # Fix the submission
            for item in submission_data:
                current_id = item["image_id"]
                # Find the correct ID
                if current_id in internal_id_to_correct_id:
                    item["image_id"] = internal_id_to_correct_id[current_id]
                    fixed_count += 1
                fixed_submission.append(item)

            print(f"Fixed {fixed_count} image IDs in submission data")

            # Save the fixed submission
            final_submission_path = os.path.join(output_dir, "submission.json")
            save_json(fixed_submission, final_submission_path)
            print(f"Saved fixed submission file to: {final_submission_path}")

        except Exception as e:
            print(f"Error fixing submission file: {e}")
            print("Falling back to original submission file")
            # Fallback: just copy the original file
            final_submission_path = os.path.join(output_dir, "submission.json")
            os.rename(expected_submission_file, final_submission_path)
            print(f"Renamed submission file to: {final_submission_path}")
    else:
        print("Please check the test evaluator configuration and work_dir.")

    # --- 5. Summary of Visualizations ---
    if args.visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        print("\nVisualization Summary:")
        print(f"All visualizations saved to: {vis_dir}")
        print("Generated visualizations include:")
        print("  - Dataset examples with ground truth masks")
        print("  - Class distribution and image size statistics")
        if not args.skip_train:
            print("  - Training loss curves and validation metrics")
        print("  - Model predictions on test images")

    print("Pipeline finished!")


if __name__ == "__main__":
    main()
