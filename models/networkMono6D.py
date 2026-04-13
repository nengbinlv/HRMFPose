#==虽然边缘预测效果有所提升，但关键点预测效果不足

import torch
from torch import nn
from models.resnet import resnet18
import torch.nn.functional as F

class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)

class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride,
                 use_bs=True
                 ):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x
# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)

class ContourPose(torch.nn.Module):
    def __init__(self,
                 fcdim=256, s16dim=256, s8dim=128, s4dim=64, s2dim=64, raw_dim=64,
                 seg_dim=2, feature_dim=64, heatmap_dim=8, edge_dim=1, graph_dim=132,
                 cat=True, dropout=0.1, sigma=100):
        super(ContourPose, self).__init__()

        self.sigma = sigma

        self.alpha = 1.0
        self.gamma = 2
        self.cat = cat  # Ture
        self.dropout = dropout  # 0.1
        self.img_convs = torch.nn.ModuleList()
        self.seg_dim = seg_dim  # 2
        self.feature_dim = feature_dim  # 64
        self.heatmap_dim = heatmap_dim
        self.edge_dim = edge_dim
        self.graph_dim = graph_dim

        self.loss_fn = nn.MSELoss()  #
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.state_loss = nn.CrossEntropyLoss()
        # self.vis_loss = nn.CrossEntropyLoss()
        self.vis_loss = nn.CrossEntropyLoss()

        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=16,
                               remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer 修改resnet全连接层
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False), # 3, 1, 1
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # The second encoder
        # x16s -> 256
        self.conv16s = nn.Sequential(
            nn.Conv2d(256 + fcdim, s16dim, 3, 1, 1, bias=False),  # 3, 1, 1
            nn.BatchNorm2d(s16dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up16sto8s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + s16dim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_raw = nn.Sequential(
            # input channel
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
        )

#=====================================================================================
        self.conv16s_2 = nn.Sequential(
            nn.Conv2d(256 + fcdim, s16dim, 3, 1, 1, bias=False),  # 3, 1, 1
            nn.BatchNorm2d(s16dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up16sto8s_2 = nn.UpsamplingBilinear2d(scale_factor=2)

        # x8s->128
        self.conv8s_2 = nn.Sequential(
            nn.Conv2d(128 + s16dim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up8sto4s_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s_2 = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )

        # x2s->64
        self.conv2s_2 = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up4sto2s_2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_raw_2 = nn.Sequential(
            # input channel
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
        )

        self.up2storaw_2 = nn.UpsamplingBilinear2d(scale_factor=2)
#=========================================================================

        self.conv_heatmap = nn.Sequential(
            nn.Conv2d(raw_dim, heatmap_dim, 1, 1)
        )


        self.conv_edge = nn.Sequential(
            nn.Conv2d(raw_dim, edge_dim, 1, 1)
        )



        self.conv_mask = nn.Sequential(
            nn.Conv2d(raw_dim, edge_dim, 1, 1)
        )


        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.classifier = nn.Linear(fcdim, 20)

        self.conv_vis = nn.Sequential(
            nn.Conv2d(fcdim, heatmap_dim, 3, 1, 1, bias=False), # 3, 1, 1
            nn.BatchNorm2d(heatmap_dim),
            nn.LeakyReLU(0.1, True)
        )

        self.avgpool_vis = nn.AdaptiveAvgPool2d((1, 1))
        self.keypoints_num = heatmap_dim

        # 这里的30是卷积后的结果
        # self.classifier_vis = nn.Linear(9600, self.keypoints_num * 2)

        self.classifier_vis_1 = nn.Linear(32, 2)
        self.classifier_vis_2 = nn.Linear(32, 2)
        self.classifier_vis_3 = nn.Linear(32, 2)
        self.classifier_vis_4 = nn.Linear(32, 2)
        self.classifier_vis_5 = nn.Linear(32, 2)
        self.classifier_vis_6 = nn.Linear(32, 2)
        self.classifier_vis_7 = nn.Linear(32, 2)
        self.classifier_vis_8 = nn.Linear(32, 2)
        # self.classifier_vis_9 = nn.Linear(32, 2)

        self.conv_graph = nn.Sequential(
            nn.Conv2d(raw_dim, graph_dim, 1, 1)
        )
        self.SE_edge = SE_Block(raw_dim)
        # self.SE_heatmap = SE_Block(raw_dim)
        # self.SE_grah = SE_Block(raw_dim)
        # self.SE_mask = SE_Block(raw_dim)

        # self.conv_edge_f1 = nn.Sequential(
        #     # input channel
        #     nn.Conv2d(raw_dim, raw_dim, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(raw_dim),
        #     nn.LeakyReLU(0.1, True),
        # )
        # self.conv_edge_b1 = nn.Sequential(
        #     nn.Conv2d(edge_dim, edge_dim, 1, 1, bias=False),
        #     nn.BatchNorm2d(1)
        # )

    # 权重交叉熵损失
    def weighted_cross_entropy_loss(self, pred_contour, target):
        """ Calculate sum of weighted cross entropy loss. """
        mask = (target > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        weight = torch.zeros_like(mask)
        weight.masked_scatter_(target > 0.5,
                               torch.ones_like(target) * num_neg / (num_pos + num_neg))
        weight.masked_scatter_(target <= 0.5,
                               torch.ones_like(target) * num_pos / (num_pos + num_neg))
        losses = F.binary_cross_entropy_with_logits(
            pred_contour.float(), target.float(), weight=weight, reduction='none')
        loss = torch.sum(losses) / b
        return loss

    def masked_smooth_l1_loss(self, map_pred, map_target, mask, sigma=1.0, normalize=True, reduce=True):
        # based on: https://github.com/zju3dv/pvnet/blob/master/lib/utils/net_utils.py
        bs, c = map_pred.shape[:2]
        sigma_2 = sigma ** 2
        map_diff = map_pred - map_target
        diff = mask * map_diff
        abs_diff = torch.abs(diff)
        sign = (abs_diff < 1. / sigma_2).detach().float()
        in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * sign + \
                (abs_diff - (0.5 / sigma_2)) * (1. - sign)
        if normalize:
            in_loss = torch.sum(in_loss.view(bs, -1), 1) / (c * torch.sum(mask.view(bs, -1), 1) + 1e-3)
        if reduce:
            in_loss = torch.mean(in_loss)
        return in_loss

    def focal_loss(self, inputs, targets, reduction = 'mean', alpha = 0.2, gamma=2.0):
        # inputs = F.sigmoid(inputs)
        # ce_loss  = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets.long())
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss

        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

    def edge_focal_loss(self, inputs, targets, reduction = 'mean',alpha = 0.1, gamma=2.0):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        # 计算概率，使用sigmoid将logits转换为概率值
        probs = torch.sigmoid(inputs)
        # 计算p_t，对于正样本就是预测为正类的概率，对于负样本就是1减去预测为正类的概率
        p_t = targets * probs + (1 - targets) * (1 - probs)
        # 计算alpha因子，根据真实标签来决定正样本和负样本的alpha权重
        alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
        # 计算调制因子 (1 - p_t) ** gamma
        modulating_factor = (1 - p_t) ** gamma
        # 最终的Focal Loss计算公式
        focal_loss = alpha_factor * modulating_factor * bce_loss
        # 求平均得到最终的损失值
        return focal_loss.mean()

    def attention_loss2(self, output, target):
        num_pos = torch.sum(target == 1).float()
        num_neg = torch.sum(target == 0).float()
        alpha = num_neg / (num_pos + num_neg) * 1.0
        eps = 1e-14
        p_clip = torch.clamp(output, min=eps, max=1.0 - eps)

        weight = target * alpha * (4 ** ((1.0 - p_clip) ** 0.5)) + \
                 (1.0 - target) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))
        weight = weight.detach()

        loss = F.binary_cross_entropy_with_logits(output, target, weight, reduction='none')
        loss = torch.mean(loss)

        return loss

    def bdcn_loss2(self, inputs, targets, l_weight=1.0):
        # bdcn loss with the rcf approach
        targets = targets.long()
        # mask = (targets > 0.1).float()
        mask = targets.float()
        num_positive = torch.sum((mask > 0.0).float()).float()  # >0.1
        num_negative = torch.sum((mask <= 0.0).float()).float()  # <= 0.1

        mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)  # 0.1
        mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
        # mask[mask == 2] = 0
        inputs = torch.sigmoid(inputs)
        cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
        # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
        cost = torch.sum(cost.float().mean((1, 2, 3)))  # before sum
        return l_weight * cost
    # 正向传播
    def forward(self, x, heatmap = None, target_contour = None, target_kpVis = None, target_graph = None,
                target_mask = None, target_mask_dilate = None, weight = None):
        # 先做一个resnet_18s:backbone
        # encoder
        # print(x)
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)
        # decoder
        # 注意这里只是decoder部分。encoder已经在resnet18_8s中包含了
        #==================关键点热图=======================
        fm1 = self.conv16s(torch.cat([xfc, x16s], 1))
        fm1 = self.up16sto8s(fm1)
        if fm1.shape[2] == 68:
            fm1 = nn.functional.interpolate(fm1, (67, 80), mode='bilinear', align_corners=False)
        fm1 = self.conv8s(torch.cat([fm1, x8s], 1))
        fm1 = self.up8sto4s(fm1)

        fm1 = self.conv4s(torch.cat([fm1, x4s], 1))
        fm1 = self.up4sto2s(fm1)

        fm1 = self.conv2s(torch.cat([fm1, x2s], 1))
        fm1 = self.up2storaw(fm1)
        if fm1.shape[2] == 536:
            fm1 = nn.functional.interpolate(fm1, (535, 640), mode='bilinear', align_corners=False)
        fm1 = self.conv_raw(torch.cat([fm1, x], 1))
        # fm1 = self.SE_heatmap(fm1)
        fm1 = self.conv_heatmap(fm1)

        pred_heatmap = fm1

        #================边缘=========================
        fm2 = self.conv16s(torch.cat([xfc, x16s], 1))
        fm2 = self.up16sto8s(fm2)
        if fm2.shape[2] == 68:
            fm2 = nn.functional.interpolate(fm2, (67, 80), mode='bilinear', align_corners=False)
        # out_1 = self.up_block_1(fm2)

        fm2 = self.conv8s(torch.cat([fm2, x8s], 1))
        fm2 = self.up8sto4s(fm2)

        fm2 = self.conv4s(torch.cat([fm2, x4s], 1))
        fm2 = self.up4sto2s(fm2)
        #3
        # out_3 = self.up_block_3(fm2)

        fm2 = self.conv2s(torch.cat([fm2, x2s], 1))
        fm2 = self.up2storaw(fm2)
        if fm2.shape[2] == 536:
            fm2 = nn.functional.interpolate(fm2, (535, 640), mode='bilinear', align_corners=False)

        fm2 = self.conv_raw(torch.cat([fm2, x], 1))
        # 加通道注意力
        fm2 = self.SE_edge(fm2)
        # fm2 = self.conv_edge_f1(fm2)
        fm2 = self.conv_edge(fm2)
        # fm2 = self.conv_edge_b1(fm2)
        pred_contour = fm2

        #=================状态==================
        # 状态分类
        # fm3 = self.avgpool(xfc)
        # # 展平
        # fm3 = fm3.view(fm3.size(0), -1)
        # fm3 = self.classifier(fm3)
        # pre_state = fm3
        #=============关键点可视化预测============
        # fm4 = self.conv_vis(xfc)

        # 换成平均池化
        fm4 = self.avgpool(xfc)
        fm4 = fm4.view(fm4.size(0), 8, -1)
        # 分为8个
        fm_v1 = self.classifier_vis_1(fm4[:,:1,:].view(fm4.size(0), -1))
        fm_v2 = self.classifier_vis_2(fm4[:, 1:2, :].view(fm4.size(0), -1))
        fm_v3 = self.classifier_vis_3(fm4[:, 2:3, :].view(fm4.size(0), -1))
        fm_v4 = self.classifier_vis_4(fm4[:, 3:4, :].view(fm4.size(0), -1))
        fm_v5 = self.classifier_vis_5(fm4[:, 4:5, :].view(fm4.size(0), -1))
        fm_v6 = self.classifier_vis_6(fm4[:, 5:6, :].view(fm4.size(0), -1))
        fm_v7 = self.classifier_vis_7(fm4[:, 6:7, :].view(fm4.size(0), -1))
        fm_v8 = self.classifier_vis_8(fm4[:, 7:8, :].view(fm4.size(0), -1))
        # fm_v9 = self.classifier_vis_8(fm4[:, 8:, :].view(fm4.size(0), -1))

        pred_vis_v1 = fm_v1
        pred_vis_v2 = fm_v2
        pred_vis_v3 = fm_v3
        pred_vis_v4 = fm_v4
        pred_vis_v5 = fm_v5
        pred_vis_v6 = fm_v6
        pred_vis_v7 = fm_v7
        pred_vis_v8 = fm_v8

        pred_vis = {}
        pred_vis["pred_vis_v1"] = pred_vis_v1
        pred_vis["pred_vis_v2"] = pred_vis_v2
        pred_vis["pred_vis_v3"] = pred_vis_v3
        pred_vis["pred_vis_v4"] = pred_vis_v4
        pred_vis["pred_vis_v5"] = pred_vis_v5
        pred_vis["pred_vis_v6"] = pred_vis_v6
        pred_vis["pred_vis_v7"] = pred_vis_v7
        pred_vis["pred_vis_v8"] = pred_vis_v8
        # pred_vis["pred_vis_v9"] = pred_vis_v9

        fm5 = self.conv16s_2(torch.cat([xfc, x16s], 1))
        fm5 = self.up16sto8s_2(fm5)
        if fm5.shape[2] == 68:
            fm5 = nn.functional.interpolate(fm5, (67, 80), mode='bilinear', align_corners=False)
        fm5 = self.conv8s_2(torch.cat([fm5, x8s], 1))
        fm5 = self.up8sto4s_2(fm5)

        fm5 = self.conv4s_2(torch.cat([fm5, x4s], 1))
        fm5 = self.up4sto2s_2(fm5)

        fm5 = self.conv2s_2(torch.cat([fm5, x2s], 1))
        fm5 = self.up2storaw_2(fm5)
        if fm5.shape[2] == 536:
            fm5 = nn.functional.interpolate(fm5, (535, 640), mode='bilinear', align_corners=False)

        fm5 = self.conv_raw_2(torch.cat([fm5, x], 1))
        # fm5 = self.SE_grah(fm5)
        fm5 = self.conv_graph(fm5)
        pred_graph = fm5

        # ================mask========================
        fm6= self.conv16s_2(torch.cat([xfc, x16s], 1))
        fm6 = self.up16sto8s_2(fm6)
        if fm6.shape[2] == 68:
            fm6 = nn.functional.interpolate(fm6, (67, 80), mode='bilinear', align_corners=False)

        fm6 = self.conv8s_2(torch.cat([fm6, x8s], 1))
        fm6 = self.up8sto4s_2(fm6)

        fm6 = self.conv4s_2(torch.cat([fm6, x4s], 1))
        fm6 = self.up4sto2s_2(fm6)

        fm6 = self.conv2s_2(torch.cat([fm6, x2s], 1))
        fm6 = self.up2storaw_2(fm6)
        if fm6.shape[2] == 536:
            fm6 = nn.functional.interpolate(fm6, (535, 640), mode='bilinear', align_corners=False)

        fm6 = self.conv_raw_2(torch.cat([fm6, x], 1))
        # fm6 = self.SE_mask(fm6)
        fm6 = self.conv_mask(fm6)

        pre_mask = fm6

        if self.training:
            loss_fn = nn.MSELoss()
            heatmap_loss = loss_fn(pred_heatmap, heatmap)
            # contour_loss = self.weighted_cross_entropy_loss(pred_contour.float(), target_contour.float())
            # contour_loss = self.attention_loss2(pred_contour.float(), target_contour.float())
            contour_loss = self.seg_loss(pred_contour.float(), target_contour.float())
            # contour_loss = self.bdcn_loss2(pred_contour.float(), target_contour.float())

            # state_loss = self.state_loss(pre_state, target_sate.long())
            # 计算每个关键点的分类损失
            vis_loss = 0.0
            vis_loss += self.vis_loss(pred_vis_v1.float(), target_kpVis[:, 0].long())
            vis_loss += self.vis_loss(pred_vis_v2.float(), target_kpVis[:, 1].long())
            vis_loss += self.vis_loss(pred_vis_v3.float(), target_kpVis[:, 2].long())
            vis_loss += self.vis_loss(pred_vis_v4.float(), target_kpVis[:, 3].long())
            vis_loss += self.vis_loss(pred_vis_v5.float(), target_kpVis[:, 4].long())
            vis_loss += self.vis_loss(pred_vis_v6.float(), target_kpVis[:, 5].long())
            vis_loss += self.vis_loss(pred_vis_v7.float(), target_kpVis[:, 6].long())
            vis_loss += self.vis_loss(pred_vis_v8.float(), target_kpVis[:, 7].long())
            # vis_loss += self.vis_loss(pred_vis_v9, target_kpVis[:, 8].long())
            vis_loss = vis_loss / 8

            graph_loss = self.masked_smooth_l1_loss(pred_graph, target_graph, target_mask)

            mask_loss = self.seg_loss(pre_mask.float(), target_mask.float())

            loss = {}
            # 任务重要性：关键点 --》 边缘 --》 mask --》 可视性 --》 边缘图
            loss["heatmap_loss"] = heatmap_loss * 1000.0 # 1000
            loss["contour_loss"] = contour_loss  * 10.0 # 10.0
            # loss["state_loss"] = state_loss
            loss["vis_loss"] = vis_loss *  0.1  #
            loss["grah_loss"] = graph_loss * 0.01  #0.01
            loss["mask_loss"] = mask_loss * 1.0

            return loss
        else:
            return pred_heatmap, pred_contour, pred_vis, pred_graph, pre_mask
