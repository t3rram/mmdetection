_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
classes = ('aerosol','conserve_ronde','conserve_rectangulaire','canette','sirops','opercule')
dataset_type = 'CocoDataset'
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))
# use caffe img_norm
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[113.17535275429276, 109.78111749183367, 105.40623458850041], std=[39.347940445996116, 39.59295708662895, 42.763824054082214], to_rgb=True)
img_norm_cfg_val = dict(
    mean=[115.02672793005365, 111.35386303303115, 109.18681316279135], std=[45.528143309429666, 47.08965165834122, 53.22657568267244], to_rgb=True)
img_norm_cfg_test = dict(
    mean=[118.94313684559191, 104.71390017541455, 84.25821877735217], std=[46.49535895192859, 47.07302265266059, 53.19300086699925], to_rgb=True)

    
 
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Corrupt', corruption = 'gaussian_blur'),
    # dict(type='CutOut', n_holes = 4, cutout_shape=(120,120)),
    # dict(type='RandomAffine'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=val_pipeline),
    test=dict(pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
