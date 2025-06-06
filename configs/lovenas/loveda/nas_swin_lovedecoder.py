import torch.nn as nn
from configs.base.loveda import train, test, data, optimizer, learning_rate
from module.nas.nasdecoder import parallel_block


config = dict(
    model=dict(
        type='NasNet',
        params=dict(
            encoder=dict(
                swin=dict(
                    pretrain_img_size=224,
                    embed_dims=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                    window_size=7,
                    use_abs_pos_embed=False,
                    drop_path_rate=0.3,
                    patch_norm=True,
                    patch_size=4,
                    mlp_ratio=4,
                    strides=(4, 2, 2, 2),
                    out_indices=(0, 1, 2, 3),
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    act_cfg=dict(type='GELU'),
                    norm_cfg=dict(type='LN'),
                    with_cp=False,
                    pretrained='/kaggle/input/swim_base_lvn/pytorch/default/1/swinbase_loveda_30k.pth',
                    frozen_stages=-1,
                    init_cfg=None
                )
            ),
            nas_decoder=dict(
                in_strides=(4, 8, 16, 32),
                in_channels=(128, 256, 512, 1024),
                channels=128,
                stacked_nums=2,
                normalized_fusion='fast_normalize',
                cell_fn=parallel_block
            ),
            head=dict(
                in_channels=128,
                out_channels=64,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
            ),

            classes=7,
            loss=dict(
                ce=dict()
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)
