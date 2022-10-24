import os
import argparse
import torch
from networks.vnet_sdf import VNet
from utils.test_patch_sass import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/omnisky/postgraduate/Yb/data_set/LASet/data', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='vnet', help='exp_name')
parser.add_argument('--model', type=str,  default='vnet_DTC', help='model_name')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--labelnum', type=int, default=11, help='labeled data')
parser.add_argument('--iter', type=int,  default=6000, help='model iteration')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')


FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/{}".format(FLAGS.model)

num_classes = 2

test_save_path = "model/{}_{}_{}_labeled/{}_predictions/".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
              image_list]


def test_calculate_metric(epoch_num):
    net = VNet(n_channels=1, n_classes=num_classes-1, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = 'model/LA_vnet_12_labeled/sassnet_label12/iter_5200_dice_0.8954771273472677.pth'
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                           save_result=False, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric(FLAGS.iter) #6000
    print(metric)

# python test_LA.py --model 0214_re01 --gpu 0
