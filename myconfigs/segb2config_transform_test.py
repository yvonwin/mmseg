# b2
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=checkpoint,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

# model training and testing settings
train_cfg=dict(),
test_cfg=dict(mode='whole' )

# dataset settings
dataset_type = 'CustomDataset'
data_root = './mmseg_data/'
classes = ['BG', 'FTU']
palette = [[0,0,0], [255,0,0]]
img_norm_cfg = dict(mean=[0.82829495,0.80269746,0.82058063], std=[0.16030526,0.18857197,0.17771745], to_rgb=True)
size = 768


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(size, size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),

    dict(
            type='Albu',
            transforms=[
                # dict(type='RandomBrightnessContrast', p=0.5),
                dict(
                    type='OneOf',
                    transforms=[
                        dict(
                            type='ElasticTransform',
                            alpha=120,
                            sigma=6.0,
                            alpha_affine=3.5999999999999996,
                            p=1),
                        dict(type='GridDistortion', p=1),
                        dict(
                            type='OpticalDistortion',
                            distort_limit=2,
                            shift_limit=0.5,
                            p=1)
                    ],
                    p=0.3),
                dict(
                    type='ShiftScaleRotate',
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=10,
                    interpolation=1,
                    p=0.5),
                # dict(type='Resize', height=640, width=640, always_apply=True, p=1),
                # dict(type='RandomCrop', height=448, width=448, p=1)
            ]),
    dict(type='MyPhotoDistortion'),
    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(size, size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    # samples_per_gpu=12,
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="splits/fold_0.txt",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="splits/valid_0.txt",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        img_dir='test/images',
        ann_dir='test/labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

total_iters = 100000
# optimizer
# optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')

optimizer_config = dict()

# learning policy
# lr_config = dict(policy='poly',
#                  warmup='linear',
#                  warmup_iters=500,
#                  warmup_ratio=1e-6,
#                  power=1.0, min_lr=0.0, by_epoch=False)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


# runtime settings
find_unused_parameters = True
runner = dict(type = 'IterBasedRunner', max_iters = total_iters)
# checkpoint_config = dict(by_epoch=False, interval=-1, save_optimizer=False)
checkpoint_config = dict(by_epoch=False, interval=10000)

# evaluation = dict(by_epoch=False, interval=500, metric='mDice', pre_eval=True)
evaluation = dict(by_epoch=False, interval=10000, metric='mDice', pre_eval=True)
fp16 = dict()
work_dir = './mmseg-mit-b2'