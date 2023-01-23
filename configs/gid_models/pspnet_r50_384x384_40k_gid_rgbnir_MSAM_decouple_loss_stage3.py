from mmseg.models.losses.decouple_loss import decouple_loss


norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DualBranchEncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone_rgb=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    backbone_nir=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=1,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    attention_module=dict(type='MSAM', plan=0),
    # the input_feature_dims of our module
    latent_convert=dict(type='LatentConvert', decouple_stage_list=[3], input_feature_dims=[2048]),
    decouple_loss=dict(type="DecoupleLoss"),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )
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
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
