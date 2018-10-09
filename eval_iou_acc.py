import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# replace the 2018-10-09-19-32-17 by your folder name
all_pred_data_label_filelist = os.path.join(BASE_DIR, 'log_ptn/inference/2018-10-09-19-32-17/result/output_filelist.txt')
pred_data_label_filenames = [line.rstrip() for line in open(all_pred_data_label_filelist)]
gt_label_filenames = [f.rstrip('_pred\.txt') + '_gt.txt' for f in pred_data_label_filenames]
num_room = len(gt_label_filenames)

gt_classes = [0 for _ in range(13)]
positive_classes = [0 for _ in range(13)]
true_positive_classes = [0 for _ in range(13)]
for i in range(num_room):
    print(i)
    data_label = np.loadtxt(pred_data_label_filenames[i])
    pred_label = data_label[:,-1]
    gt_label = np.loadtxt(gt_label_filenames[i])
    print(gt_label.shape)
    for j in range(gt_label.shape[0]):
        gt_l = int(gt_label[j])
        pred_l = int(pred_label[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l==pred_l)

print(gt_classes)
print(positive_classes)
print(true_positive_classes)

print('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))

print('IoU:')
iou_list = []
for i in range(13):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
    print(iou)
    iou_list.append(iou)
print('mIOU: ', sum(iou_list)/13.0)

print('Acc:')
acc_list = []
for i in range(13):
    acc = true_positive_classes[i]/float(gt_classes[i])
    print(acc)
    acc_list.append(acc)
print('mAcc:', sum(acc_list)/13.0)