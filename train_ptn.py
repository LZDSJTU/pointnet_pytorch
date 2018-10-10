from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import argparse
from datetime import datetime

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.autograd import Variable

from model.pointnet import PointNet
from utils.train_utils import AverageMeter, accuracy, adjust_learning_rate, shuffle_data, loadDataFile, getDataFiles

def main():
    parser = argparse.ArgumentParser(description='Voxelnet for semantic')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--epochs', default=100, help='epochs')
    parser.add_argument('--batchsize', default=32, help='epochs')
    parser.add_argument('--weight_file', default='', help='weights to load')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 6]')
    parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')

    args = parser.parse_args()
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batchsize
    lr = args.lr
    ALL_FILES = getDataFiles('indoor3d_sem_seg_hdf5_data/all_files.txt')
    room_filelist = [line.rstrip() for line in open('indoor3d_sem_seg_hdf5_data/room_filelist.txt')]

    # Load ALL data
    data_batch_list = []
    label_batch_list = []
    for h5_filename in ALL_FILES:
        data_batch, label_batch = loadDataFile(h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    print(data_batches.shape)
    print(label_batches.shape)

    test_area = 'Area_' + str(args.test_area)
    train_idxs = []
    test_idxs = []
    for i, room_name in enumerate(room_filelist):
        if test_area in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)

    train_data = data_batches[train_idxs, ...]
    train_label = label_batches[train_idxs].astype(np.int64)
    test_data = data_batches[test_idxs, ...]
    test_label = label_batches[test_idxs].astype(np.int64)
    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = os.path.join('log_ptn/train', test_area + '_' + time_string)
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    checkpoint_dir = os.path.join(log_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    writer = SummaryWriter(log_dir=os.path.join( log_dir, 'tensorboard'))

    start_epoch = 0
    epochs = args.epochs

    model = get_model()
    model.cuda()
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.weight_file != '':
        pre_trained_model = torch.load(args.weight_file)
        start_epoch = pre_trained_model['epoch']
        model_state = model.state_dict()
        model_state.update(pre_trained_model['state_dict'])
        model.load_state_dict(model_state)

    global_counter = 0
    for epoch in range(start_epoch, epochs):
        learn_rate_now = adjust_learning_rate(optimizer, global_counter, BATCH_SIZE, lr)
        writer.add_scalar('train/learning_rate', learn_rate_now, global_counter)

        losses = AverageMeter()
        top1 = AverageMeter()
        model.train()

        train_data_shuffled, train_label_shuffled, _ = shuffle_data(train_data[:, 0:NUM_POINT, :], train_label)
        file_size = train_data_shuffled.shape[0]
        num_batches = file_size // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            feature = train_data_shuffled[start_idx:end_idx, :, :]
            label = train_label_shuffled[start_idx:end_idx]

            feature = np.expand_dims(feature, axis=1)
            input = Variable(torch.from_numpy(feature).cuda(), requires_grad=True)
            input = torch.transpose(input, 3, 1)
            target = Variable(torch.from_numpy(label).cuda(), requires_grad=False)
            target = target.view(-1,)
            output = model(input)
            output_reshaped = output.permute(0, 3, 2, 1).contiguous().view(-1, 13)

            loss = criterion(output_reshaped, target)
            prec1 = accuracy(output_reshaped.data, target.data, topk=(1,))
            prec1[0] = prec1[0].cpu().numpy()[0]
            losses.update(loss.data[0], BATCH_SIZE)
            top1.update(prec1[0], BATCH_SIZE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch: [{0}][{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, batch_idx, loss=losses, top1=top1))

            with open(os.path.join(log_dir,'train_log.txt'), 'a') as f:
                f.write('Epoch: [{0}][{1}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n'.format(
                    epoch, batch_idx, loss=losses, top1=top1))

            global_counter += 1

        writer.add_scalar('train/loss', losses.avg, global_counter)
        writer.add_scalar('train/accuracy', top1.avg, global_counter)


        losses = AverageMeter()
        top1 = AverageMeter()
        model.eval()

        file_size = test_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            feature = test_data[start_idx:end_idx, :, :]
            label = test_label[start_idx:end_idx]

            feature = np.expand_dims(feature, axis=1)
            input = Variable(torch.from_numpy(feature).cuda(), requires_grad=True)
            input = torch.transpose(input, 3, 1)
            target = Variable(torch.from_numpy(label).cuda(), requires_grad=False)
            target = target.view(-1,)
            output = model(input)
            output_reshaped = output.permute(0, 3, 2, 1).contiguous().view(-1, 13)

            loss = criterion(output_reshaped, target)
            prec1 = accuracy(output_reshaped.data, target.data, topk=(1,))
            prec1[0] = prec1[0].cpu().numpy()[0]
            losses.update(loss.data[0], BATCH_SIZE)
            top1.update(prec1[0], BATCH_SIZE)

        writer.add_scalar('val/loss', losses.avg, global_counter)
        writer.add_scalar('val/accuracy', top1.avg, global_counter)

        print('Epoch {} Val Loss {:.3f} Val Acc {:.3f}  \t'
              .format(epoch, losses.avg, top1.avg))

        with open(os.path.join(log_dir, 'test_log.txt'), 'a') as f:
            f.write('Epoch: [{0}]\t'
                    'Loss {loss.avg:.4f} \t'
                    'Prec@1 {top1.avg:.3f} \n'.format(
                epoch, loss=losses, top1=top1))

        if(epoch % 5 == 0):
            torch.save(
                {'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pth.tar') )

    writer.close()

def get_model():
    model = PointNet()
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    return model

if __name__ == '__main__':
    main()