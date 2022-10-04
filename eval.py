import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from loss import Loss,cal_dice
from dataloaders.la_heart import LAHeart, CenterCrop, ToTensor
from networks.vnet import VNet


def eval_loop(model, criterion, valid_loader, device):
    model.eval()
    running_loss = 0
    dice_valid = 0

    with torch.no_grad():
        for sampled_batch in valid_loader:
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            outputs = model(volume_batch)

            loss = criterion(outputs, label_batch)
            dice = cal_dice(outputs, label_batch)
            print('dice: {:.3f}'.format(dice))
            running_loss += loss.item()
            dice_valid += dice.item()

    loss = running_loss / len(valid_loader)
    dice = dice_valid / len(valid_loader)
    return {'loss': loss, 'dice': dice}


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = '/***、data_set/LASet/data'
patch_size = (112,112,80)
model = VNet(n_channels=1,n_classes=2, normalization='batchnorm').to(device)
# 加载训练模型
weight_path = 'results/VNet.pth'
weight_dict = torch.load(weight_path, map_location=device)
model.load_state_dict(weight_dict)
print('Successfully loading checkpoint.')
criterion = Loss(n_classes=2).to(device)
db_test = LAHeart(base_dir=data_path,
                        split='test',
                        transform=transforms.Compose([
                        CenterCrop(patch_size),
                        ToTensor()
                    ]))
testloader = DataLoader(db_test,batch_size=1, num_workers=4, pin_memory=True)

model.eval()
valid_metrics = eval_loop(model, criterion, testloader, device)
# 这里的dice是测试集中心裁剪的dice
dice = valid_metrics['dice']
print('Average dice: {:.5f}'.format(dice))
