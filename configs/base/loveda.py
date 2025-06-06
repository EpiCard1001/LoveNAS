from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
import ever as er
from ever.preprocess.albu import RandomDiscreteScale

data = dict(
    train=dict(
        type='LoveDALoader',
        params=dict(
            image_dir=[
                '/kaggle/input/loveda-dataset/Train/Train/Rural/images_png',
                '/kaggle/input/loveda-dataset/Train/Train/Urban/images_png/',
            ],
            mask_dir=[
                '/kaggle/input/loveda-dataset/Train/Train/Rural/masks_png',
                '/kaggle/input/loveda-dataset/Train/Train/Urban/masks_png',
            ],
            transforms=Compose([
                RandomDiscreteScale([0.5, 0.75, 1.0, 1.25, 1.5, 1.75]),
                RandomCrop(512, 512),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ]),
            CV=dict(k=10, i=-1),
            training=True,
            batch_size=16,
            num_workers=4,
        ),
    ),
    test=dict(
        type='LoveDALoader',
        params=dict(
            image_dir=[
                '/kaggle/working/datalns/Urban/images_png',
                '/kaggle/working/datalns/Rural/images_png',
            ],
            mask_dir=[
                '/kaggle/working/datalns/Urban/masks_png',
                '/kaggle/working/datalns/Rural/masks_png',
            ],
            transforms=Compose([
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ]),
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=2,
            num_workers=2,
        ),
    ),
)


optimizer = dict(
    type='sgd',
    params=dict(
        momentum=0.9,
        weight_decay=0.0001
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)


learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.01,
        power=0.9,
        max_iters=15000,
    ))
train = dict(
    forward_times=1,
    num_iters=15000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=30,
    eval_interval_epoch=30,
)

test = dict(

)
