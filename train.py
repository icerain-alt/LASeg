import os
import torch
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.vnet import VNet
from loss import Loss,cal_dice
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor


def train_loop(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0
    pbar = tqdm(train_loader)
    dice_train = 0

    for sampled_batch in pbar:
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        # print(volume_batch.shape,label_batch.shape)
        outputs = model(volume_batch)
        # print(outputs.shape)
        loss = criterion(outputs, label_batch)
        dice = cal_dice(outputs, label_batch)
        dice_train += dice.item()
        pbar.set_postfix(loss="{:.3f}".format(loss.item()), dice="{:.3f}".format(dice.item()))

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = running_loss / len(train_loader)
    dice = dice_train / len(train_loader)
    return {'loss': loss, 'dice': dice}


def eval_loop(model, criterion, valid_loader, device):
    model.eval()
    running_loss = 0
    pbar = tqdm(valid_loader)
    dice_valid = 0

    with torch.no_grad():
        for sampled_batch in pbar:
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            outputs = model(volume_batch)

            loss = criterion(outputs, label_batch)
            dice = cal_dice(outputs, label_batch)
            running_loss += loss.item()
            dice_valid += dice.item()
            pbar.set_postfix(loss="{:.3f}".format(loss.item()), dice="{:.3f}".format(dice.item()))

    loss = running_loss / len(valid_loader)
    dice = dice_valid / len(valid_loader)
    return {'loss': loss, 'dice': dice}


def train(args, model, optimizer, criterion, train_loader, valid_loader, epochs,
          device, train_log, loss_min=999.0):
    for e in range(epochs):
        # train for epoch
        train_metrics = train_loop(model, optimizer, criterion, train_loader, device)
        valid_metrics = eval_loop(model, criterion, valid_loader, device)

        # eval for epoch
        info1 = "Epoch:[{}/{}] train_loss: {:.3f} valid_loss: {:.3f}".format(e + 1, epochs, train_metrics["loss"],
                                                                             valid_metrics['loss'])
        info2 = "train_dice: {:.3f} valid_dice: {:.3f}".format(train_metrics['dice'], valid_metrics['dice'])

        print(info1 + '\n' + info2)
        with open(train_log, 'a') as f:
            f.write(info1 + '\n' + info2 + '\n')

        if valid_metrics['loss'] < loss_min:
            loss_min = valid_metrics['loss']
            torch.save(model.state_dict(), args.save_path)
    print("Finished Training!")


def main(args):
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data info
    db_train = LAHeart(base_dir=args.train_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(args.patch_size),
                               ToTensor(),
                           ]))
    db_test = LAHeart(base_dir=args.train_path,
                          split='test',
                          transform=transforms.Compose([
                              CenterCrop(args.patch_size),
                              ToTensor()
                          ]))
    print('Using {} images for training, {} images for testing.'.format(len(db_train), len(db_test)))
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=True)
    testloader = DataLoader(db_test, batch_size=1, num_workers=4, pin_memory=True)
    model = VNet(n_channels=1,n_classes=args.num_classes, normalization='batchnorm', has_dropout=True).to(device)

    criterion = Loss(n_classes=args.num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=1e-4)

    # 加载训练模型
    if os.path.exists(args.weight_path):
        weight_dict = torch.load(args.weight_path, map_location=device)
        model.load_state_dict(weight_dict)
        print('Successfully loading checkpoint.')

    train(args, model, optimizer, criterion, trainloader, testloader, args.epochs, device, train_log=args.train_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--patch_size', type=float, default=(112, 112, 80))
    parser.add_argument('--train_path', type=str, default='/***data_set/LASet/data')
    parser.add_argument('--train_log', type=str, default='results/VNet_sup.txt')
    parser.add_argument('--weight_path', type=str, default='results/VNet_sup.pth')  # 加载
    parser.add_argument('--save_path', type=str, default='results/VNet_sup.pth')  # 保存
    args = parser.parse_args()

    main(args)