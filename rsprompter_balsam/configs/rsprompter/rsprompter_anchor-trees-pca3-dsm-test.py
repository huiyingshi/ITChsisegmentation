"""PCA RGB + DSM test config; leaves the HSI-adapter experiment unchanged."""

_base_ = ['./rsprompter_anchor-trees-hsi-dsm.py']

# Keep this test run isolated from the HSI-adapter checkpoints and logs.
work_dir = '/home/huiying/Documents/ITCsegmentation/Code/Output/balsam_pca3_dsm_test'
resume = False
load_from = None
crop_size = (1024, 1024)

# PCA PNGs are RGB uint8 images. These are SAM's ImageNet RGB statistics,
# expressed on the 0-255 scale expected by the model's data preprocessor.
rgb_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
rgb_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
data_preprocessor = dict(
    type='DSMDetDataPreprocessor',
    mean=rgb_mean,
    std=rgb_std,
    dsm_norm='max',
    bgr_to_rgb=False,  # LoadImageFromFile is explicitly asked for RGB order.
    pad_mask=False,
    pad_size_divisor=32,
)

# Reuse the existing detector and SAM backbone; only bypass the spectral adapter.
model = dict(
    hsi_adapter=None,
    data_preprocessor=data_preprocessor,
)

# The inherited visualization hook reads original HSI bands, so disable it for
# this PCA RGB experiment. Standard MMEngine evaluation remains enabled.
custom_hooks = []

data_root = '/home/huiying/Documents/ITCsegmentation/Data/DatasetNew/processed/pca_1024_stride256_png'
dataset_type = 'TreesHSIDSMDataset'

batch_size_per_gpu = 1
num_workers = 0
persistent_workers = False

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, channel_order='rgb'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadDSMFromRasterio', to_float32=True),
    # Keep PCA RGB, DSM and masks geometrically synchronized during training.
    dict(type='HSIDSMRandomGeoAug',
         horizontal_prob=0.5,
         vertical_prob=0.5,
         scale_prob=0.7,
         scale_range=(0.85, 1.15)),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'dsm_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, channel_order='rgb'),
    dict(type='LoadDSMFromRasterio', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'dsm_path', 'ori_shape', 'img_shape')),
]

train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
