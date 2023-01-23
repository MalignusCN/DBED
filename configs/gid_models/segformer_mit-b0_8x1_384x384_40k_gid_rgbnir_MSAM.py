# _base_ = [
#     '../_base_/models/segformer_mit-b0.py',
#     '../_base_/datasets/cityscapes_1024x1024.py',
#     '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
# ]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DualBranchEncoderDecoder',
    pretrained=None,
    backbone_rgb=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    backbone_nir=dict(
        type='MixVisionTransformer',
        in_channels=1,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    attention_module=dict(type='MSAM', plan=0),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
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
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)


