# 추론 전용 ViTPose config (학습 설정 없음)

# _base_ = ['../../../_base_/default_runtime.py']  # logging 등 기본 설정만 유지

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# 모델 설정
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch={
            'embed_dims': 384,
            'num_layers': 12,
            'num_heads': 12,
            'feedforward_channels': 384 * 4
        },
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.1,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_small_20230913.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=384,
        out_channels=17,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),  # ✅ 삭제 금지
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False
    ),
)
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='TopDownCocoDataset',
        ann_file='',
        data_prefix=dict(img=''),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(256, 192)),
            dict(type='PackPoseInputs')
        ],
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
    )


