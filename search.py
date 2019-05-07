# -*- coding: utf-8 -*-
# @Time    : 2018-12-10
# @Author  : huangbofan
import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
from PIL import Image
import tqdm
import matplotlib
import time

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')


class PULPDataset(Dataset):

    def __init__(self, list_path, img_base_folder, transform=None):
        imgs = []
        self.train_labels = []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                str = json.loads(line)
                img_path = os.path.join(img_base_folder, str['url'])
                imgs.append((img_path, str['label']))
                self.train_labels.append(str['label'])
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        file, label = self.imgs[index]
        img = Image.open(file).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index

    def __len__(self):
        return len(self.imgs)


def search(net, base_Features, preprocessed_test_img, K, base_Labels=None):
    net.eval()
    with torch.no_grad():
        inputs = preprocessed_test_img.cuda()
        features = net(inputs)
        dist = torch.mm(features, base_Features)
        _, yi = dist.topk(K, dim=1, largest=True, sorted=True)

        # candidates = trainLabels.view(1, -1).expand(batchSize, -1)
        # retrieval = torch.gather(candidates, 1, yi)
        #
        # retrieval_one_hot.resize_(batchSize * K, C).zero_()
        # retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        # yd_transform = yd.clone().div_(sigma).exp_()
        # probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)),
        #                   1)
        # _, predictions = probs.sort(1, True)
        #
        # # Find which predictions match the target
        # correct = predictions.eq(targets.data.view(-1, 1))
        # cls_time.update(time.time() - end)
        #
        # top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        # top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        #
        # total += targets.size(0)
        #
        # print('Test [{}/{}]\t'
        #       'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
        #       'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
        #       'Top1: {:.2f}  Top5: {:.2f}'.format(
        #     total, testsize, top1 * 100. / total, top5 * 100. / total, net_time=net_time, cls_time=cls_time))

    return yi.cpu().numpy().tolist()[0]


def recompute():
    pass


def main(args):
    # model
    tic = time.time()
    model = torchvision.models.resnet18(False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 128)
    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lemniscate = checkpoint['lemniscate']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        return

    print("=> MODEL LOADING TIME: {:.6f}s".format(time.time()-tic))
    tic = time.time()

    if args.img:
        img = Image.open(args.img).convert("RGB")
    else:
        print('no img')
        return

    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("=> IMG PREPOSESSING TIME: {:.6f}s".format(time.time()-tic))
    tic = time.time()

    if (args.list_path is None) != (args.img_base_folder is None):
        print('Error: --list_path and --img_base_folder must both be supplied or omitted')
        return
    else:
        list_path = args.list_path
        img_base_folder = args.img_base_folder
    train_dataset = PULPDataset(list_path=list_path, img_base_folder=img_base_folder, transform=transform_img)

    trainFeatures = lemniscate.memory.t()

    print("=> BASE DATASET LOADING TIME: {:.6f}s".format(time.time()-tic))
    tic = time.time()

    # 不是默认，需要重新计算获得 memory
    if args.list_path != '/workspace/mnt/data/train/samples/index.lst':
        print('recompute_memory...')
        model.eval()
        temploader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False, num_workers=2)
        for batch_idx, (inputs, targets, indexes) in enumerate(tqdm.tqdm(temploader)):
            batchSize = inputs.size(0)
            features = model(inputs)
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
        # base_Labels = torch.LongTensor(temploader.dataset.train_labels).cuda()

    base_Features = trainFeatures
    preprocessed_img = transform_img(img)
    preprocessed_img = preprocessed_img.unsqueeze_(0)

    yi = search(net=model, base_Features=base_Features, preprocessed_test_img=preprocessed_img, K=args.number)

    print("=> SEARCHING TIME: {:.6f}s".format(time.time()-tic))
    tic = time.time()

    out_path = './output/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    size = (500, 500)
    org_img = Image.open(args.img).convert('RGB').resize(size, Image.ANTIALIAS)
    plt.figure(figsize=(60, 6))
    plt.suptitle('result')  # 图片标图
    G = gridspec.GridSpec(1, args.number + 1, wspace=0.1)
    plt.subplot(G[0]), plt.title('search img', fontsize=10)
    plt.imshow(org_img), plt.axis('off')

    cunt = 1
    for index in yi:
        path, _ = train_dataset.imgs[index]
        img = Image.open(path).convert('RGB').resize(size, Image.ANTIALIAS)
        plt.subplot(G[cunt]), plt.title(os.path.basename(path), fontsize=10)
        plt.imshow(img), plt.axis('off')
        cunt += 1
    # plt.tight_layout()
    plt_figure = os.path.join(out_path, os.path.basename(args.img))
    plt.savefig(plt_figure)

    print("=> PLOTTING TIME: {:.6f}s".format(time.time()-tic))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--img', default='', required=True, type=str, help='path to img')
    parser.add_argument('--model', default='', required=True, type=str, help='path to model')
    parser.add_argument('--number', default='10', type=int)
    parser.add_argument('--list_path', '-l', default='/workspace/mnt/data/train/samples/index.lst', type=str,
                        help='path to img list file')
    parser.add_argument('--img_base_folder', '-i', default='/workspace/mnt/data/train/', type=str,
                        help='path to img base_folder')
    args = parser.parse_args()
    main(args)

