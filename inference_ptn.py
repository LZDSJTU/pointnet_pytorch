from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable

from model.pointnet import PointNet
from utils.train_utils import AverageMeter, accuracy
from utils.indoor3d_util import room2blocks_wrapper_normalized, g_label2color

parser = argparse.ArgumentParser(description='Voxelnet for semantic')
parser.add_argument('--batchsize', default=1, help='epochs')
parser.add_argument('--weight_file', default='', help='weights to load')
parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 6]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--room_data_filelist', required=True, help='TXT filename, filelist, each line is a test room data label file.')
parser.add_argument('--output_filelist', required=True, help='TXT filename, filelist, each line is an output for a room')
parser.add_argument('--dump_dir', required=True, help='dump folder path')
parser.add_argument('--visu', action='store_true', help='Whether to output OBJ file for prediction visualization.')

args = parser.parse_args()

time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
log_dir = os.path.join('log_ptn/inference', time_string)
if not os.path.exists(log_dir): os.makedirs(log_dir)

NUM_POINT = args.num_point
BATCH_SIZE = args.batchsize
ROOM_PATH_LIST = [os.path.join(BASE_DIR + '/data_preparation/s3dis_npy', line.rstrip()) for line in open(args.room_data_filelist)]
DUMP_DIR = args.dump_dir
DUMP_DIR = os.path.join(log_dir, DUMP_DIR)
if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR)
OUTPUT_FILELIST = os.path.join(log_dir,args.output_filelist)
NUM_CLASSES = 13

def get_model():
    model = PointNet()
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    return model

def evaluate(room_path, out_data_label_filename, out_gt_label_filename):
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    if args.visu:
        fout = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4] + '_pred.obj'), 'w')
        fout_gt = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4] + '_gt.obj'), 'w')
    fout_data_label = open(out_data_label_filename, 'w')
    fout_gt_label = open(out_gt_label_filename, 'w')

    current_data, current_label = room2blocks_wrapper_normalized(room_path, NUM_POINT)
    current_data = current_data[:, 0:NUM_POINT, :].astype(np.float32)
    current_label = np.squeeze(current_label).astype(np.int64)

    # Get room dimension..
    data_label = np.load(room_path)
    data = data_label[:, 0:6]
    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx

        feature = current_data[start_idx:end_idx, :, :]
        label = current_label[start_idx:end_idx]

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

        pred_label = np.reshape(np.argmax(output_reshaped.data.cpu().numpy(), axis=1), (BATCH_SIZE,-1))
        pred_val = np.reshape(output_reshaped.data.cpu().numpy(), (BATCH_SIZE,-1,13))

        # Save prediction labels to OBJ file
        for b in range(BATCH_SIZE):
            pts = current_data[start_idx + b, :, :]
            l = current_label[start_idx + b, :]
            pts[:, 6] *= max_room_x
            pts[:, 7] *= max_room_y
            pts[:, 8] *= max_room_z
            pts[:, 3:6] *= 255.0
            pred = pred_label[b, :]
            for i in range(NUM_POINT):
                color = g_label2color[pred[i]]
                color_gt = g_label2color[current_label[start_idx + b, i]]
                if args.visu:
                    fout.write(
                        'v %f %f %f %d %d %d\n' % (pts[i, 6], pts[i, 7], pts[i, 8], color[0], color[1], color[2]))
                    fout_gt.write('v %f %f %f %d %d %d\n' % (
                        pts[i, 6], pts[i, 7], pts[i, 8], color_gt[0], color_gt[1], color_gt[2]))
                fout_data_label.write('%f %f %f %d %d %d %f %d\n' % (pts[i, 6], pts[i, 7], pts[i, 8], pts[i, 3], pts[i, 4], pts[i, 5], pred_val[b, i, pred[i]], pred[i]))
                fout_gt_label.write('%d\n' % (l[i]))
        correct = np.sum(pred_label == current_label[start_idx:end_idx, :])
        total_correct += correct
        total_seen += (cur_batch_size * NUM_POINT)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[i - start_idx, j] == l)

    print('eval accuracy: %f' % (total_correct / float(total_seen)))
    fout_data_label.close()
    fout_gt_label.close()
    if args.visu:
        fout.close()
        fout_gt.close()

    return total_correct, total_seen

model = get_model()
model.cuda()
criterion = nn.CrossEntropyLoss().cuda()

if args.weight_file != '':
    pre_trained_model = torch.load(args.weight_file)
    model_state = model.state_dict()
    model_state.update(pre_trained_model['state_dict'])
    model.load_state_dict(model_state)

total_correct = 0
total_seen = 0
fout_out_filelist = open(OUTPUT_FILELIST, 'w')
for room_path in ROOM_PATH_LIST:
    out_data_label_filename = os.path.basename(room_path)[:-4] + '_pred.txt'
    out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
    out_gt_label_filename = os.path.basename(room_path)[:-4] + '_gt.txt'
    out_gt_label_filename = os.path.join(DUMP_DIR, out_gt_label_filename)
    print(out_data_label_filename)
    a, b = evaluate(room_path, out_data_label_filename, out_gt_label_filename)
    total_correct += a
    total_seen += b
    fout_out_filelist.write(out_data_label_filename + '\n')
fout_out_filelist.close()

print('all room eval accuracy: %f' % (total_correct / float(total_seen)))

if __name__ == '__main__':
    print('finished!')
