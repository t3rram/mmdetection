_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
classes = ('aerosol','conserve_ronde','conserve_rectangulaire','canette','sirops','opercule', 'autre_acier')
dataset_type = 'CocoDataset'
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=7,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight = [1.39,0.87,2.166,0.696,1.34,0.52,1.757,1.0]),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)))
    )
# use caffe img_norm
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[113.17535275429276, 109.78111749183367, 105.40623458850041], std=[39.347940445996116, 39.59295708662895, 42.763824054082214], to_rgb=True)


    
 
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

# Modify dataset related settings

data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        pipeline=train_pipeline,
        img_prefix='/home/shared/emr/data/acier/v-2021-10-27/images/train/',
        classes=classes,
        ann_file='/home/shared/emr/data/acier/v-2021-10-27/labels/train_ann.json',
        type = 'CocoDataset'),
    val=dict(
        pipeline=val_pipeline,
        img_prefix='/home/shared/emr/data/acier/v-2021-10-27/images/val_semi_dense/',
        classes=classes,
        ann_file='/home/shared/emr/data/acier/v-2021-10-27/labels/val_semi_dense_ann.json',
        type = 'CocoDataset'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix='/home/shared/emr/data/acier/v-2021-10-27/images/test_acier/',
        classes=classes,
        ann_file='/home/shared/emr/data/acier/v-2021-10-27/labels/acier.json',
        type = 'CocoDataset'),
        )
data_root = '/home/shared/emr/data/acier/v-2021-10-27/'

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
###########################

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    warmup='linear',  # The warmup policy, also support `exp` and `constant`.
    warmup_iters=500,  # The number of iterations for warmup
    warmup_ratio=
    0.001,  # The ratio of the starting learning rate used for warmup
    step=[8, 11]
    )  # Steps to decay the learning rate
runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=2) # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=1)  # The save interval is 1
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook')
    ])  # The logger used to record the training process.
#dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
#log_level = 'INFO'  # The level of logging.

#resume_from =  "/home/shared/tools/mmdetection/r50_subclass/latest.pth" # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1), ('val', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.
work_dir = 'test'  # Directory to save the model checkpoints and logs for the current experiments.
