_base_ = ['_base_/rsprompter_anchor.py']
 
default_scope = 'mmdet'
custom_imports = dict(imports=['mmdet.rsprompter'], allow_failed_imports=False)

work_dir = '/home/huiying/Documents/ITCsegmentation/Code/Output/balsam_hsi_dsm_supervised_DataAug_1024_stride256_anchor_trees'
crop_size = (1024, 1024)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best='coco/segm_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(type='HSITestVisualizationHook', output_dir='test_predictions',
         val_output_dir='val_predictions', rgb_bands=(76, 51, 26),
         score_thr=0.3),
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')


num_classes = 1
prompt_shape = (70, 5)  # (per img pointset, per pointset point)

#### should be changed when using different pretrain model

hf_sam_pretrain_name = '/home/huiying/Documents/ITCsegmentation/Code/BalSAM/pretrain_models/sam_vit_base'
# huggingface model name, e.g. facebook/sam-vit-base
# or local repo path, e.g. work_dirs/sam_cache/sam_vit_base
hf_sam_pretrain_ckpt_path = hf_sam_pretrain_name + '/pytorch_model.bin'
data_preprocessor = dict(
    type='DSMDetDataPreprocessor',
    # HSI loading must normalize all 164 bands before this stage. The future
    # model adapter must map them to the three channels expected by SAM.
    mean=None,
    std=None,
    dsm_norm="max",
    bgr_to_rgb=False,
    pad_mask=False, #True,
    pad_size_divisor=32,

)

# model settings
model = dict(
    type='RSPrompterAnchorDSM',
    data_preprocessor=data_preprocessor,
    hsi_adapter=dict(
        type='HSISpectralAdapter',
        input_channels=164,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        mul=2,
        out_channels=3,
        pixel_chunk_size=16384,
    ),
    decoder_freeze=False,
    shared_image_embedding=dict(
        type='RSSamPositionalEmbedding',
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
    ),
    backbone=dict(
        type='RSSamVisionEncoder',
        hf_pretrain_name=hf_sam_pretrain_name,
        extra_config=dict(output_hidden_states=True),
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)),
    

    neck=dict(
        feature_aggregator=dict(
            in_channels=hf_sam_pretrain_name,
            hidden_channels=32,
            select_layers=range(1, 13, 2),  #### should be changed when using different pretrain model, base: range(1, 13, 2), large: range(1, 25, 2), huge: range(1, 33, 2)
        ),
    ),

    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
    
    roi_head=dict(
        type='RSPrompterAnchorRoIPromptHeadDSM',
        bbox_head=dict(
            num_classes=num_classes,
        ),

        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        
        mask_head=dict(
            type='RSPrompterAnchorMaskHeadDSM',
            
            mask_decoder=dict(
                type='RSSamMaskDecoder',
                hf_pretrain_name=hf_sam_pretrain_name,
                init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)),
            per_pointset_point=prompt_shape[1],
            with_sincos=True,
            dsm_modulated_image = False, 
            ),
            
        )
)


dataset_type = 'TreesHSIDSMDataset'
data_root = '/home/huiying/Documents/ITCsegmentation/Data/DatasetNew/processed/balsam_hsi_512_stride256'

# Start with a single worker and image per GPU for the first end-to-end check.
batch_size_per_gpu = 1
num_workers = 0
persistent_workers = False

# Keep all 164 HSI bands. The fixed 10000 DN scale is shared by every band
# and tile; it preserves spectral ratios and is not reflectance calibration.
# Rasterio loaders replace HSI/DSM nodata with zero. The exporter writes both
# inputs and COCO masks on an aligned 512 x 512 grid with the original footprint.
train_pipeline = [
    dict(type='LoadHSIFromFile', expected_channels=164,
         scale_factor=10000.0, nodata_value=65535),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadDSMFromRasterio', to_float32=True),
    # Synchronised geometry: both flips are independently sampled, then an
    # optional modest zoom is cropped/padded back to 512 x 512.  HSI, DSM,
    # boxes and masks always receive the same transform.
    dict(type='HSIDSMRandomGeoAug',
         horizontal_prob=0.5,
         vertical_prob=0.5,
         scale_prob=0.7,
         scale_range=(0.85, 1.15)),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'dsm_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction'))
]

test_pipeline = [
    dict(type='LoadHSIFromFile', expected_channels=164,
         scale_factor=10000.0, nodata_value=65535),
    dict(type='LoadDSMFromRasterio', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'dsm_path', 'ori_shape', 'img_shape'))
]

# _delete_ prevents WHU paths and prefixes from the inherited base config
# from leaking into this experiment. COCO file_name/dsm_path are absolute.
train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    _delete_=True,
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

find_unused_parameters = True

resume = True  # Auto-resume from the latest checkpoint in work_dir/seed_0.
load_from = None

base_lr = 0.00001
max_epochs=100

train_cfg = dict(max_epochs=max_epochs, val_interval=3)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.001,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
    )
    #dict(type="ExponentialLR", gamma=0.9)
]

backend_args = None

# The checkpoint hook monitors coco/segm_mAP from this one-class evaluator.
val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    classwise=False,
    format_only=False,
    backend_args=None,
)
test_evaluator = val_evaluator

#### AMP training config
runner_type = 'Runner'
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05)
)

randomness = dict(seed=0)

#### DeepSpeed training config
# runner_type = 'FlexibleRunner'
# strategy = dict(
#     type='DeepSpeedStrategy',
#     fp16=dict(
#         enabled=True,
#         auto_cast=False,
#         fp16_master_weights_and_grads=False,
#         loss_scale=0,
#         loss_scale_window=500,
#         hysteresis=2,
#         min_loss_scale=1,
#         initial_scale_power=15,
#     ),
#     inputs_to_half=['inputs'],
#     zero_optimization=dict(
#         stage=2,
#         allgather_partitions=True,
#         allgather_bucket_size=2e8,
#         reduce_scatter=True,
#         reduce_bucket_size='auto',
#         overlap_comm=True,
#         contiguous_gradients=True,
#     ),
# )
# optim_wrapper = dict(
#     type='DeepSpeedOptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.05
#     )
# )
