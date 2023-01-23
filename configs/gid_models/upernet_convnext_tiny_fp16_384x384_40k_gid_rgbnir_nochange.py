norm_cfg = dict(type='SyncBN', requires_grad=True)
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa
model = dict(
    type='GIDEncoderDecoderWithNIR',
    pretrained=None,
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='tiny',
        in_channels=4,
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'GaofenImageDataset'
data_root = '/home/visint-book/Dataset/GID_cropped'
img_norm_cfg = dict(
    mean=[126.596, 94.443, 99.451, 92.385], \
    std=[4.615, 5.148, 4.801, 4.901], to_rgb=False)
crop_size = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFileNPY'),
    dict(type='LoadAnnotationsNPY'),
    dict(type='Resize', img_scale=(384, 384), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(384, 384), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[126.596, 94.443, 99.451, 92.385],
        std=[4.615, 5.148, 4.801, 4.901],
        to_rgb=False),
    dict(type='Pad', size=(384, 384), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFileNPY'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 384),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[126.596, 94.443, 99.451, 92.385],
                std=[4.615, 5.148, 4.801, 4.901],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='GaofenImageDataset',
        data_root='/home/visint-book/Dataset/GID_cropped',
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=[
            dict(type='LoadImageFromFileNPY'),
            dict(type='LoadAnnotationsNPY'),
            dict(
                type='Resize', img_scale=(384, 384), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(384, 384), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            # dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[126.596, 94.443, 99.451, 92.385],
                std=[4.615, 5.148, 4.801, 4.901],
                to_rgb=False),
            dict(type='Pad', size=(384, 384), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='GaofenImageDataset',
        data_root='/home/visint-book/Dataset/GID_cropped',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=[
            dict(type='LoadImageFromFileNPY'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(384, 384),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[126.596, 94.443, 99.451, 92.385],
                        std=[4.615, 5.148, 4.801, 4.901],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='GaofenImageDataset',
        data_root='/home/visint-book/Dataset/GID_cropped',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=[
            dict(type='LoadImageFromFileNPY'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(384, 384),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[126.596, 94.443, 99.451, 92.385],
                        std=[4.615, 5.148, 4.801, 4.901],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 6
    })

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
