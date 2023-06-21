model = dict(
    type='ImageClassifier',#mmcls/models/classifiers/image.py
    backbone=dict(
        type='VisionTransformer',#mmcls/models/backbones/vision_transformer.py
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',#mmcls/models/heads/vision_transformer_head.py
        num_classes=102,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, #[0 1] * (1 — 0.1) + 0.1 / 2 =[0 1]*(0.9) + 0.05
            mode='classy_vision'),#mmcls/models/losses/label_smooth_loss.py
        hidden_dim=3072),
    train_cfg=dict(
        augments=dict(
            type='BatchMixup', alpha=0.2, num_classes=102, prob=1.0)))
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',#https://blog.csdn.net/u011583927/article/details/104724419/
        policies=[[{
            'type': 'Posterize',## 降低图片位数
            'bits': 4,
            'prob': 0.4
        }, {
            'type': 'Rotate',# 旋转
            'angle': 30.0,
            'prob': 0.6
        }],
                  [{
                      'type': 'Solarize',## 翻转部分暗色像素
                      'thr': 113.77777777777777,
                      'prob': 0.6
                  }, {
                      'type': 'AutoContrast',#图像对比度
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',#直方图均衡化
                      'prob': 0.8
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Posterize',
                      'bits': 5,
                      'prob': 0.6
                  }, {
                      'type': 'Posterize',
                      'bits': 5,
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.4
                  }, {
                      'type': 'Solarize',
                      'thr': 142.22222222222223,
                      'prob': 0.2
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.4
                  }, {
                      'type': 'Rotate',
                      'angle': 26.666666666666668,
                      'prob': 0.8
                  }],
                  [{
                      'type': 'Solarize',
                      'thr': 170.66666666666666,
                      'prob': 0.6
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Posterize',
                      'bits': 6,
                      'prob': 0.8
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 10.0,
                      'prob': 0.2
                  }, {
                      'type': 'Solarize',
                      'thr': 28.444444444444443,
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.6
                  }, {
                      'type': 'Posterize',
                      'bits': 5,
                      'prob': 0.4
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 26.666666666666668,
                      'prob': 0.8
                  }, {
                      'type': 'ColorTransform',
                      'magnitude': 0.0,
                      'prob': 0.4
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 30.0,
                      'prob': 0.4
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.0
                  }, {
                      'type': 'Equalize',
                      'prob': 0.8
                  }],
                  [{
                      'type': 'Invert',
                      'prob': 0.6
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.4,
                      'prob': 0.6
                  }, {
                      'type': 'Contrast',
                      'magnitude': 0.8,
                      'prob': 1.0
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 26.666666666666668,
                      'prob': 0.8
                  }, {
                      'type': 'ColorTransform',
                      'magnitude': 0.2,
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.8,
                      'prob': 0.8
                  }, {
                      'type': 'Solarize',
                      'thr': 56.888888888888886,
                      'prob': 0.8
                  }],
                  [{
                      'type': 'Sharpness',
                      'magnitude': 0.7,
                      'prob': 0.4
                  }, {
                      'type': 'Invert',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Shear',
                      'magnitude': 0.16666666666666666,
                      'prob': 0.6,
                      'direction': 'horizontal'
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.0,
                      'prob': 0.4
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.4
                  }, {
                      'type': 'Solarize',
                      'thr': 142.22222222222223,
                      'prob': 0.2
                  }],
                  [{
                      'type': 'Solarize',
                      'thr': 113.77777777777777,
                      'prob': 0.6
                  }, {
                      'type': 'AutoContrast',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Invert',
                      'prob': 0.6
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.4,
                      'prob': 0.6
                  }, {
                      'type': 'Contrast',
                      'magnitude': 0.8,
                      'prob': 1.0
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.8
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }]]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='MyFilelist',
        data_prefix='D:\\eclipse-workspace\\PyTorch4\\mmclassification-master\\mmcls\\data\\flower_data\\train_filelist',
        ann_file='D:\\eclipse-workspace\\PyTorch4\\mmclassification-master\\mmcls\\data\\flower_data\\train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),#mmcls/datasets/pipelines/loading.py
            dict(
                type='RandomResizedCrop',#mmcls/datasets/pipelines/transforms.py
                size=224,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='AutoAugment',#mmcls/datasets/pipelines/auto_augment.py
                policies=[[{
                    'type': 'Posterize',
                    'bits': 4,
                    'prob': 0.4
                }, {
                    'type': 'Rotate',
                    'angle': 30.0,
                    'prob': 0.6
                }],
                          [{
                              'type': 'Solarize',
                              'thr': 113.77777777777777,
                              'prob': 0.6
                          }, {
                              'type': 'AutoContrast',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.8
                          }, {
                              'type': 'Equalize',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Posterize',
                              'bits': 5,
                              'prob': 0.6
                          }, {
                              'type': 'Posterize',
                              'bits': 5,
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.4
                          }, {
                              'type': 'Solarize',
                              'thr': 142.22222222222223,
                              'prob': 0.2
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.4
                          }, {
                              'type': 'Rotate',
                              'angle': 26.666666666666668,
                              'prob': 0.8
                          }],
                          [{
                              'type': 'Solarize',
                              'thr': 170.66666666666666,
                              'prob': 0.6
                          }, {
                              'type': 'Equalize',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Posterize',
                              'bits': 6,
                              'prob': 0.8
                          }, {
                              'type': 'Equalize',
                              'prob': 1.0
                          }],
                          [{
                              'type': 'Rotate',
                              'angle': 10.0,
                              'prob': 0.2
                          }, {
                              'type': 'Solarize',
                              'thr': 28.444444444444443,
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.6
                          }, {
                              'type': 'Posterize',
                              'bits': 5,
                              'prob': 0.4
                          }],
                          [{
                              'type': 'Rotate',
                              'angle': 26.666666666666668,
                              'prob': 0.8
                          }, {
                              'type': 'ColorTransform',
                              'magnitude': 0.0,
                              'prob': 0.4
                          }],
                          [{
                              'type': 'Rotate',
                              'angle': 30.0,
                              'prob': 0.4
                          }, {
                              'type': 'Equalize',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.0
                          }, {
                              'type': 'Equalize',
                              'prob': 0.8
                          }],
                          [{
                              'type': 'Invert',
                              'prob': 0.6
                          }, {
                              'type': 'Equalize',
                              'prob': 1.0
                          }],
                          [{
                              'type': 'ColorTransform',
                              'magnitude': 0.4,
                              'prob': 0.6
                          }, {
                              'type': 'Contrast',
                              'magnitude': 0.8,
                              'prob': 1.0
                          }],
                          [{
                              'type': 'Rotate',
                              'angle': 26.666666666666668,
                              'prob': 0.8
                          }, {
                              'type': 'ColorTransform',
                              'magnitude': 0.2,
                              'prob': 1.0
                          }],
                          [{
                              'type': 'ColorTransform',
                              'magnitude': 0.8,
                              'prob': 0.8
                          }, {
                              'type': 'Solarize',
                              'thr': 56.888888888888886,
                              'prob': 0.8
                          }],
                          [{
                              'type': 'Sharpness',
                              'magnitude': 0.7,
                              'prob': 0.4
                          }, {
                              'type': 'Invert',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Shear',
                              'magnitude': 0.16666666666666666,
                              'prob': 0.6,
                              'direction': 'horizontal'
                          }, {
                              'type': 'Equalize',
                              'prob': 1.0
                          }],
                          [{
                              'type': 'ColorTransform',
                              'magnitude': 0.0,
                              'prob': 0.4
                          }, {
                              'type': 'Equalize',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.4
                          }, {
                              'type': 'Solarize',
                              'thr': 142.22222222222223,
                              'prob': 0.2
                          }],
                          [{
                              'type': 'Solarize',
                              'thr': 113.77777777777777,
                              'prob': 0.6
                          }, {
                              'type': 'AutoContrast',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Invert',
                              'prob': 0.6
                          }, {
                              'type': 'Equalize',
                              'prob': 1.0
                          }],
                          [{
                              'type': 'ColorTransform',
                              'magnitude': 0.4,
                              'prob': 0.6
                          }, {
                              'type': 'Contrast',
                              'magnitude': 0.8,
                              'prob': 1.0
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.8
                          }, {
                              'type': 'Equalize',
                              'prob': 0.6
                          }]]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='ImageNet',
        data_prefix='../mmcls/data/flower_data/val_filelist',
        ann_file='../mmcls/data/flower_data/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=(256, -1),
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='ImageNet',
        data_prefix='../mmcls/data/flower_data/val_filelist',
        ann_file='../mmcls/data/flower_data/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=(256, -1),
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy')
paramwise_cfg = dict(
    custom_keys=dict({
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }))
optimizer = dict(
    type='AdamW',
    lr=0.003,
    weight_decay=0.3,
    paramwise_cfg=dict(
        custom_keys=dict({
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })))
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=50)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/vit-base-p16_pt-64xb64_in1k-224'
gpu_ids = [0]
