import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.unet_urpc import unet_3D_dv_semi
from utils import ramps, losses
from dataloaders.la_heart import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/***/data_set/LASet/data', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='vnet', help='exp_name')
parser.add_argument('--model', type=str,  default='URPC', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int,  default=25, help='trained samples')
parser.add_argument('--max_samples', type=int, default=123, help='all samples')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

patch_size = (112, 112, 80)
snapshot_path = "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum, args.model)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)


def cal_dice(output, target, eps=1e-3):
    output = torch.argmax(output,dim=1)
    inter = torch.sum(output * target) + eps
    union = torch.sum(output) + torch.sum(target) + eps * 2
    dice = 2 * inter / union
    return dice


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
        
    model = unet_3D_dv_semi(n_classes=num_classes, in_channels=1).cuda()
    db_train = LAHeart(base_dir=args.root_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    db_test = LAHeart(base_dir=args.root_path,
                      split='test',
                      transform=transforms.Compose([
                          CenterCrop(patch_size),
                          ToTensor()
                      ]))
    
    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(db_test, batch_size=1,shuffle=False, num_workers=4, pin_memory=True)
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    kl_distance = nn.KLDivLoss(reduction='none')

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations//len(trainloader)+1
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]

            outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4,  = model(volume_batch)
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
            outputs_aux4_soft = torch.softmax(outputs_aux4, dim=1)

            loss_ce_aux1 = ce_loss(outputs_aux1[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])
            loss_ce_aux2 = ce_loss(outputs_aux2[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])
            loss_ce_aux3 = ce_loss(outputs_aux3[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])
            loss_ce_aux4 = ce_loss(outputs_aux4[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])

            loss_dice_aux1 = dice_loss(
                outputs_aux1_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux2 = dice_loss(
                outputs_aux2_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux3 = dice_loss(
                outputs_aux3_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux4 = dice_loss(
                outputs_aux4_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

            supervised_loss = (loss_ce_aux1+loss_ce_aux2+loss_ce_aux3+loss_ce_aux4 +
                               loss_dice_aux1+loss_dice_aux2+loss_dice_aux3+loss_dice_aux4)/8

            preds = (outputs_aux1_soft +
                     outputs_aux2_soft+outputs_aux3_soft+outputs_aux4_soft)/4

            variance_aux1 = torch.sum(kl_distance(
                torch.log(outputs_aux1_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux1 = torch.exp(-variance_aux1)

            variance_aux2 = torch.sum(kl_distance(
                torch.log(outputs_aux2_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(kl_distance(
                torch.log(outputs_aux3_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            variance_aux4 = torch.sum(kl_distance(
                torch.log(outputs_aux4_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux4 = torch.exp(-variance_aux4)

            consistency_dist_aux1 = (
                preds[args.labeled_bs:] - outputs_aux1_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux1 = torch.mean(
                consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

            consistency_dist_aux2 = (
                preds[args.labeled_bs:] - outputs_aux2_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux2 = torch.mean(
                consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

            consistency_dist_aux3 = (
                preds[args.labeled_bs:] - outputs_aux3_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux3 = torch.mean(
                consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

            consistency_dist_aux4 = (
                preds[args.labeled_bs:] - outputs_aux4_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux4 = torch.mean(
                consistency_dist_aux4 * exp_variance_aux4) / (torch.mean(exp_variance_aux4) + 1e-8) + torch.mean(variance_aux4)

            consistency_loss = (consistency_loss_aux1 +
                                consistency_loss_aux2 + consistency_loss_aux3 + consistency_loss_aux4) / 4
            consistency_weight = get_current_consistency_weight(iter_num//150)
            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/supervised_loss',
                              supervised_loss, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, supervised_loss: %f' %
                (iter_num, loss.item(), supervised_loss.item()))

            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                with torch.no_grad():
                    dice_sample = 0
                    for sampled_batch in test_loader:
                        img, lbl = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
                        outputs = model(img)
                        dice_once = cal_dice(outputs[0],lbl)
                        dice_sample += dice_once
                    dice_sample = dice_sample / len(test_loader)
                    print('Average center dice:{:.3f}'.format(dice_sample))
                    
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
