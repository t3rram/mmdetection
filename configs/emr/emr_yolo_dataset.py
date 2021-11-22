
# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_c4_1x_coco_emr_yolo.py'


bbox_head=dict(
            type='BBoxHead',
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=2048,
            num_classes=7,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
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
# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('aerosol','conserve_ronde','conserve_rectangulaire','canette','sirops','opercule','autre_acier')
data = dict(
    train=dict(
        img_prefix='dataset_augmented_acier/images/',
        classes=classes,
        ann_file='dataset_augmented_acier/labels/train_ann.json',
        type = 'CocoDataset'),
    val=dict(
        img_prefix='dataset_augmented_acier/images/',
        classes=classes,
        ann_file='dataset_augmented_acier/labels/val_ann.json',
        type = 'CocoDataset'),
    test=dict(
        img_prefix='dataset_augmented_acier/images/',
        classes=classes,
        ann_file='dataset_augmented_acier/labels/test_ann.json',
        type = 'CocoDataset'),
        )
data_root = 'dataset_augmented_acier/'

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    warmup='linear',  # The warmup policy, also support `exp` and `constant`.
    warmup_iters=1,  # The number of iterations for warmup
    warmup_ratio=
    0.001,  # The ratio of the starting learning rate used for warmup
    step=[8, 11]
    )  # Steps to decay the learning rate
runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=5) # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=1)  # The save interval is 1
log_config = dict(  # config to register logger hook
    interval=30,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook')
    ])  # The logger used to record the training process.
#dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
#log_level = 'INFO'  # The level of logging.

resume_from =  "/home/shared/tools/mmdetection/work_dir_/latest.pth" # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1), ('val', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.
work_dir = 'work_dir_'  # Directory to save the model checkpoints and logs for the current experiments.