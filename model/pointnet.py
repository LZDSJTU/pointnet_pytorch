import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        # input: B, 6, N, P

        self.conv_1 = nn.Conv2d(9, 64, kernel_size=(1,1), stride=(1,1))
        # self.bn_1 = nn.BatchNorm2d(64, momentum=0.5)
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1,1))
        # self.bn_2 = nn.BatchNorm2d(64, momentum=0.5)
        self.bn_2 = nn.BatchNorm2d(64)

        self.conv_3 = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1,1))
        # self.bn_3 = nn.BatchNorm2d(64, momentum=0.5)
        self.bn_3 = nn.BatchNorm2d(64)

        self.conv_4 = nn.Conv2d(64, 128, kernel_size=(1,1), stride=(1,1))
        # self.bn_4 = nn.BatchNorm2d(128, momentum=0.5)
        self.bn_4 = nn.BatchNorm2d(128)

        self.conv_5 = nn.Conv2d(128, 1024, kernel_size=(1,1), stride=(1,1))
        # self.bn_5 = nn.BatchNorm2d(1024, momentum=0.5)
        self.bn_5 = nn.BatchNorm2d(1024)

        self.conv_6 = nn.Conv2d(1152, 512, kernel_size=(1,1), stride=(1,1))
        self.bn_6 = nn.BatchNorm2d(512)

        self.conv_7 = nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1))
        self.bn_7 = nn.BatchNorm2d(256)

        self.dp = nn.Dropout(p=0.3)

        self.conv_8 = nn.Conv2d(256, 13, kernel_size=(1,1), stride=(1,1))

        self.global_conv_1 = nn.Conv2d(1024, 256, kernel_size=(1,1), stride=(1,1))
        # self.globa_bn_1 = nn.BatchNorm2d(256, momentum=0.5)
        self.globa_bn_1 = nn.BatchNorm2d(256)

        self.global_conv_2 = nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1,1))
        # self.globa_bn_2 = nn.BatchNorm2d(128, momentum=0.5)
        self.globa_bn_2 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=0.3)


    def forward(self, input):

        _, _, point_num, _ = input.size()

        conv_1 = self.relu(self.bn_1(self.conv_1(input)))
        conv_2 = self.relu(self.bn_2(self.conv_2(conv_1)))
        conv_3 = self.relu(self.bn_3(self.conv_3(conv_2)))
        conv_4 = self.relu(self.bn_4(self.conv_4(conv_3)))
        conv_5 = self.relu(self.bn_5(self.conv_5(conv_4)))

        global_feature = F.max_pool2d(conv_5, (point_num, 1))
        global_feature = self.relu(self.globa_bn_1(self.global_conv_1(global_feature)))
        global_feature = self.relu(self.globa_bn_2(self.global_conv_2(global_feature)))
        global_feature_repeat = global_feature.repeat(1, 1, point_num, 1)

        points_feat_concat = torch.cat((conv_5, global_feature_repeat), 1)

        conv_6 = self.relu(self.bn_6(self.conv_6(points_feat_concat)))
        conv_7 = self.relu(self.bn_7(self.conv_7(conv_6)))

        droped = self.dp(conv_7)
        conv_8 = self.conv_8(droped)

        # conv_2 = self.relu(nn.BatchNorm2d(64, momentum=0.5)(self.conv_2(conv_1)))
        # conv_3 = self.relu(nn.BatchNorm2d(64, momentum=0.5)(self.conv_3(conv_2)))
        # conv_4 = self.relu(nn.BatchNorm2d(128, momentum=0.5)(self.conv_4(conv_3)))
        # conv_5 = self.relu(nn.BatchNorm2d(1024, momentum=0.5)(self.conv_5(conv_4)))
        #
        # global_feature = F.max_pool2d(conv_5, (point_num, 1))
        # global_feature = self.relu(nn.BatchNorm2d(256, momentum=0.5)(self.global_conv_1(global_feature)))
        # global_feature = self.relu(nn.BatchNorm2d(128, momentum=0.5)(self.global_conv_2(global_feature)))
        # global_feature_repeat = global_feature.repeat(1, 1, point_num, 1)
        #
        # points_feat_concat = torch.cat((conv_5, global_feature_repeat), 1)
        #
        # conv_6 = self.relu(nn.BatchNorm2d(512, momentum=0.5)(self.conv_6(points_feat_concat)))
        # conv_7 = self.relu(nn.BatchNorm2d(256, momentum=0.5)(self.conv_7(conv_6)))


        return conv_8








