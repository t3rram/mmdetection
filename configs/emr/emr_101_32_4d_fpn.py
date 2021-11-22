
# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco.py'


# use caffe img_norm
img_norm_cfg = dict(
    mean=[114.317, 110.594, 106.789], std=[36.972, 37.5339, 40.898], to_rgb=True)

# policies = [
#              [
#                  dict(
#                      type='Shear',
#                      prob=0.4,
#                      level=0,
#                      axis='x')
#              ],
#              [
#                  dict(
#                      type='Rotate',
#                      prob=0.6,
#                      level=10)
#              ],             
#              [
#                  dict(
#                      type='Translate',
#                      prob=0.6,
#                      level=3)
#              ],
#              [
#                  dict(
#                      type='EqualizeTransform',
#                      prob=0.5)
#              ]
#          ]

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Corrupt', corruption = 'gaussian_blur'),
    dict(type='CutOut', n_holes = 4, cutout_shape=(120,120)),
    #dict(type='Mosaic', img_scale=(1280, 1280)),
    dict(type='RandomAffine'),
    #dict(type='RandomCrop', crop_size = (608,608)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            #dict(type='RandomAffine'),
            #dict(type='RandomCrop', crop_size = (608,608)),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
    


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
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))


model = dict(
    roi_head=dict(
        bbox_head=bbox_head
    ))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('aerosol','conserve_ronde','conserve_rectangulaire','canette','sirops','opercule','autre_acier')
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='dataset_augmented_acier/images/',
        classes=classes,
        ann_file='dataset_augmented_acier/labels/train_ann.json',
        type = 'CocoDataset',
        pipeline=train_pipeline),
    val=dict(
        img_prefix='dataset_augmented_acier/images/',
        classes=classes,
        ann_file='dataset_augmented_acier/labels/test_ann.json',
        type = 'CocoDataset',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='dataset_augmented_acier/images/',
        classes=classes,
        ann_file='dataset_augmented_acier/labels/test_ann.json',
        type = 'CocoDataset',
        pipeline=test_pipeline),
        )
data_root = 'dataset_augmented_acier/'

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/faster_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.412_20200506_041400-64a12c0b.pth'

# lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
#     policy='step',  # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
#     warmup='linear',  # The warmup policy, also support `exp` and `constant`.
#     warmup_iters=100,  # The number of iterations for warmup
#     warmup_ratio=
#     0.001,  # The ratio of the starting learning rate used for warmup
#     step=[8, 11]
#     )  # Steps to decay the learning rate
runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=35) # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
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

#resume_from =  "/home/shared/tools/mmdetection/work_dir_/latest.pth" # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1), ('val', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.
work_dir = '101_fpn_35epoch_valdense'  # Directory to save the model checkpoints and logs for the current experiments.
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(_delete_=True, type='Adam', lr=0.0001, eps=1e-08, weight_decay=0.0001)