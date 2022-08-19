# b2 单分类pb 0.71
# TODO TEST

num_classes = 6

norm_cfg = dict(type='SyncBN', requires_grad=True)
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'  # noqa
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
)
# model training and testing settings
train_cfg=dict(),
test_cfg=dict(mode='whole' )

# dataset settings
dataset_type = 'CustomDataset'
data_root = './mmseg_multi_data/'
classes = ['background', 'kidney', 'prostate', 'largeintestine', 'spleen', 'lung']
palette = [[0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255]]

img_norm_cfg = dict(mean=[0,0,0], std=[1,1,1], to_rgb=True)
size = 768


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(size, size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),

    dict(
            type='Albu',
            transforms=[
                # dict(type='RandomBrightnessContrast', p=0.5),
                dict(type='RandomRotate90', p=1),
                # dict(
                #     type='OneOf',
                #     transforms=[
                #         dict(
                #             type='ElasticTransform',
                #             alpha=120,
                #             sigma=6.0,
                #             alpha_affine=3.5999999999999996,
                #             p=1),
                #         dict(type='GridDistortion', p=1),
                #         dict(
                #             type='OpticalDistortion',
                #             distort_limit=2,
                #             shift_limit=0.5,
                #             p=1)
                #     ],
                #     p=0.3),
                dict(
                    type='ShiftScaleRotate',
                    shift_limit=0,
                    scale_limit=(0.5, 2),
                    rotate_limit=(-45, 45),
                    interpolation=1,
                    border_mode=0,
                    value = (0,0,0),
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir='masks',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="splits/fold_0.txt",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir='masks',
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
        img_dir='train',
        ann_dir='masks',
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


# optimizer
# optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
# optimizer
optimizer = dict(
    type='AdamW',
    # lr=0.00006,
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'pos_block': dict(decay_mult=0.),
    #         'norm': dict(decay_mult=0.),
    #         'head': dict(lr_mult=10.)
    #     })
    )

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')

# optimizer_config = dict()

# learning policy
# lr_config = dict(policy='poly',
#                  warmup='linear',
#                  warmup_iters=500,
#                  warmup_ratio=1e-6,
#                  power=1.0, min_lr=0.0, by_epoch=False)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

total_iters = 80000
# runtime settings
find_unused_parameters = True
runner = dict(type = 'IterBasedRunner', max_iters = total_iters)
# checkpoint_config = dict(by_epoch=False, interval=-1, save_optimizer=False)
checkpoint_config = dict(by_epoch=False, interval=4000)

# evaluation = dict(by_epoch=False, interval=500, metric='mDice', pre_eval=True)
evaluation = dict(by_epoch=False, interval=4000, metric='mDice', pre_eval=True)
fp16 = dict()
work_dir = './convnext_uper_multi'