import os
from pickle import TRUE
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
from model.ourmodel import NTRENet   
from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0) 

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')
parser.add_argument('--save', default='test', type=str, help='define the name head for model storing')
parser.add_argument('-weight', default=5.0, type=float, help='aux_weight')
parser.add_argument('-method', default="ResNet50", type=str, help='name for the used backbone')

path_choice = {
    'ResNet50': 'results/pascal/resnet50_1shot/btz_12/train_epoch_90_0.6093789180381786.pth',
    'ResNet101': 'results/pascal/resnet101_1shot/1shots/train_epoch_64_0.6153272755219727.pth',
    'VGG-16': 'results/pascal/vgg_1shot/1shots/train_epoch_58_0.54536345754956.pth',
}

def get_parser():
    cfg = config.load_cfg_from_cfg_file('config/pascal_split0_resnet50.yaml')
    args = parser.parse_args()
    cfg.aux_weight = args.weight
    cfg.save_path = args.save
    cfg.method = args.method
    return cfg


def get_logger(logger_name):
    # logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    
    ### multi-processing training is deprecated
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False    # sync_bn is deprecated 
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    BatchNorm = nn.BatchNorm2d

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = NTRENet(layers=args.layers, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=BatchNorm, \
        pretrained=True, shot=args.shot, ppm_scales=args.ppm_scales, vgg=args.vgg,bgpro_num = args.bgpro_num)
    
    optimizer = model._optimizer(args)


    global logger, writer
    writer = SummaryWriter(args.save_path)
    # logger = get_logger(os.path.join(writer.logdir, 'training.log'))
    logging.basicConfig(filename=os.path.join(writer.logdir, 'training.log'), level=logging.DEBUG)
    logging.info("=> creating model ...")
    logging.info("Classes: {}".format(args.classes))
    logging.info(model)
    print(args)

    model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            logging.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded weight '{}'".format(args.weight))
        else:
            logging.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))


    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]
    train_transform = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, \
                                data_list=args.train_list, transform=train_transform, mode='train', \
                                use_coco=args.use_coco, use_split_coco=args.use_split_coco)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])    
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])           
        val_data = dataset.SemData_return_original_image(split=args.split, shot=args.shot, data_root=args.data_root, \
                                data_list=args.val_list, transform=val_transform, mode='val', \
                                use_coco=args.use_coco, use_split_coco=args.use_split_coco)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    max_iou = 0.
    filename = 'NTRENet.pth'

    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)
        if epoch ==0:
            prototype_neg_dict = dict()
        epoch_log = epoch + 1

        if args.evaluate and (epoch % 2 == 0 or (args.epochs<=50 and epoch%1==0)):
            validate_save(val_loader, model, criterion, train_data.sub_list)

    filename = args.save_path + '/final.pth'
    logging.info('Saving checkpoint to: ' + filename)
    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)                


def validate_save(val_loader, model, criterion, base_list):
    if main_process():
        logging.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    base_list = [i-1 for i in base_list]

    if args.use_coco:
        split_gap = 20
    else:
        split_gap = 5
    class_intersection_meter = [0]*split_gap
    class_union_meter = [0]*split_gap  

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    end = time.time()
    if args.split != 999:
        if args.use_coco:
            test_num = 20000
        else:
            test_num = 1000 
    else:
        test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0    
    iter_num = 0
    total_time = 0
    for e in range(10):
        for i, (input, target, s_input, s_mask, subcls, ori_label, ori_image) in enumerate(val_loader):
            if (iter_num-1) * args.batch_size_val >= test_num:
                break
            iter_num += 1    
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            start_time = time.time()
            output = model(s_x=s_input, s_y=s_mask, x=input, y=target,classes = subcls, prototype_neg_dict=None)

            total_time = total_time + 1
            model_time.update(time.time() - start_time)

            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda()*255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)         
            loss = criterion(output, target)    

            n = input.size(0)
            loss = torch.mean(loss)

            output = output.max(1)[1]

            write_fig(ori_label.cpu().numpy()[0], output.cpu().numpy()[0], ori_image.cpu().numpy()[0], i)
            if i>=7:
                break
        break
    return


def write_fig(target, segmentation_result, original_image, index):
    class_colors = [
    [0, 0, 0],    # Background (black)
    [255, 0, 0],  # Class 1 (red)
    # [0, 255, 0],  # Class 2 (green)
    # [0, 0, 255],  # Class 3 (blue)
    # Add more colors if you have more classes
    ]
    
    segmentation_result = cv2.resize(segmentation_result.astype('float'), (original_image.shape[1], original_image.shape[0]))
    visualization = np.zeros([original_image.shape[0], original_image.shape[1], 3], dtype=float)
    visualization_ground_truth = np.zeros([original_image.shape[0], original_image.shape[1], 3], dtype=float)

    for class_index in range(len(class_colors)):
        mask = segmentation_result == class_index
        visualization[mask] = class_colors[class_index]

        mask_gt = target == class_index
        visualization_ground_truth[mask_gt] = class_colors[class_index]

    alpha = 0.5  # Adjust the transparency
    result = cv2.addWeighted(original_image, 1 - alpha, visualization, alpha, 0, dtype=cv2.CV_32F)

    result_gt =  cv2.addWeighted(original_image, 1 - alpha, visualization_ground_truth, alpha, 0, dtype=cv2.CV_32F)

    cv2.imwrite(f"try/original_image_{index}.jpg", original_image)
    cv2.imwrite(f"try/predicted_image_{index}.jpg", result)
    cv2.imwrite(f'try/ground_truth_image_{index}.jpg', result_gt)

if __name__ == '__main__':
    main()
