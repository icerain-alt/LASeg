import os
import sys
from sklearn.utils import shuffle
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import time
import random

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.vnet import VNet
from utils import ramps, losses
from dataloaders.la_heart import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/omnisky/postgraduate/Yb/data_set/LASet/data',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='vnet', help='model_name')
parser.add_argument('--model', type=str, default='supervised', help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=25, help='trained samples')
parser.add_argument('--max_samples', type=int, default=123, help='all samples')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
args = parser.parse_args()


num_classes = 2
patch_size = (112, 112, 80)
snapshot_path = "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum, args.model)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def cal_dice(output, target, eps=1e-3):
    output = torch.argmax(output,dim=1)
    inter = torch.sum(output * target) + eps
    union = torch.sum(output) + torch.sum(target) + eps * 2
    dice = 2 * inter / union
    return dice


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    db_train = LAHeart(base_dir=args.root_path,
                       split='train',
                       num=args.labelnum,
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


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=args.batch_size,shuffle=True, num_workers=4, pin_memory=True,drop_last=True,
                              worker_init_fn=worker_init_fn)
    test_loader = DataLoader(db_test, batch_size=1,shuffle=False, num_workers=4, pin_memory=True)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(train_loader)))

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(train_loader) + 1
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(train_loader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)

            # calculate the loss
            loss_seg = F.cross_entropy(outputs, label_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
            loss = 0.5 * (loss_seg + loss_seg_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)

            logging.info('iteration %d : loss : %f ' % (iter_num, loss.item()))

            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                with torch.no_grad():
                    dice_sample = 0
                    for sampled_batch in test_loader:
                        img, lbl = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
                        outputs = model(img)
                        dice_once = cal_dice(outputs,lbl)
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
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
