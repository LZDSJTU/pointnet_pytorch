import numpy as np
import math
import h5py

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, global_counter, batch_size, base_lr):
    lr = max(base_lr * (0.5 ** (global_counter*batch_size // 300000)), 1e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_bn_decay(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    bn_momentum = 0.5 * (0.5 ** (epoch // 10))
    bn_decay = np.minimum(1-bn_momentum, 0.99)
    return bn_decay

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def loadDataFile(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]