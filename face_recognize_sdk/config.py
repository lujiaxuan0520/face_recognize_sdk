import torch
import torchvision.transforms as T

class Config:
    # network settings
    backbone = 'resnet' # [resnet, fmobile]
    metric = 'arcface'  # [cosface, arcface]
    embedding_size = 512
    drop_ratio = 0.5

    # data preprocess
    input_shape = [1, 128, 128]
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomHorizontalFlip(),
        T.Resize((144, 144)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    # dataset
    # train_root = 'data/CASIA-WebFace_masked'
    train_root = 'data/test-mask-slim'
    test_root = "data/lfw-align-128"
    test_list = "data/lfw_test_pair.txt"
    # test_root = "data/masked_whn"
    # test_list = "data/masked_whn_pairs_new.txt"
    # test_root = "data/test-mask-slim"
    # test_list = "data/test-mask-slim-pair.txt"
    
    # training settings
    checkpoints = "checkpoints/webface-ResIRSE-best" # webface-ResIRSE or test-mask-slim or fmobile
    restore = False
    restore_model = "95.pth"
    test_model = "checkpoints/webface-ResIRSE-best/96_5.25.pth"
    # test_model = "checkpoints/webface-ResIRSE_masked/52_10.47.pth"
    # test_model = "checkpoints/test-mask-slim/79_10.86.pth"
    
    train_batch_size = 128 # 64
    test_batch_size = 60 # 60

    epoch = 100 # 24
    optimizer = 'adam'  # ['sgd', 'adam']
    # lr = 1e-1
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss' # ['focal_loss', 'cross_entropy']
    gpu_devices = "5"

    pin_memory = True  # if memory is large, set it True to speed up a bit
    num_workers = 4  # dataloader

config = Config()
