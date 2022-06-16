# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector
import torch.nn as nn
import torchvision
import numpy as np
import random
import numpy as np
from PIL import Image
import pdb
import math
import cv2
import imageio


@DETECTORS.register_module()
class VoteNet(SingleStage3DDetector):
    r"""`VoteNet <https://arxiv.org/pdf/1904.09664.pdf>`_ for 3D detection."""

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoteNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)
        # Encoder, Fusion and Decoder
        self.stairnet = StairNet(5)  # inputs: [b, 5, h, w]
        self.img_encoder = Img_Encoder()  # inputs: [b, 3, h, w]
        self.decoder = Decoder()  # inputs:[fusion, d13_o1]

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img=None,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        batch_size = len(points)
        range_H16 = []
        range_H32 = []
        range_H64 = []
        position = []
        crop_image = []
        
        # print("img_metas:{}".format(img_metas))
        for i in range(batch_size):
            # print(-1.0 * img_metas[i]['pcd_rotation'][1][0])
            # print(points[i])
            # print(points[i].shape)
            points[i] = self.points_rotation(points[i], np.arcsin(img_metas[i]['pcd_rotation'][1][0]))
            H16, H32, H64, p = self.lidar_to_range_gpu(points[i])
            range_H16.append(H16.unsqueeze(0))
            range_H32.append(H32.unsqueeze(0))
            range_H64.append(H64.unsqueeze(0))
            position.append(p)
            

        range_H16 = torch.cat(range_H16, dim=0)
        range_H32 = torch.cat(range_H32, dim=0)
        range_H64 = torch.cat(range_H64, dim=0)
        crop_image = F.interpolate(img[:, :, :, :], (64, 512), mode='bilinear')
        # print(range_H16.shape)
        range16 = range_H16.cpu().numpy()
        range16 = range16[0,0,:,:]  
        # print(range16.shape)
        imageio.imsave("test.png",range16.astype(np.uint8))


        crop_image = add_noise(crop_image)
        image = crop_image.cpu().numpy()
        image1 = image[0,:,:,:]
        # print(image1.shape)

        image2 = image1.transpose(1,2,0)
        # print(np.transpose(image1,(1,2,0)).shape)
        
        # print(image2.shape)
        # cv2.imwrite("image2.jpg", image2)
        
        range_feature_16, range_feature_32, range_feature_64 = self.stairnet([range_H16, range_H32, range_H64])
        img_feature = self.img_encoder(crop_image)
        # print(img_feature.shape)
        # image_feature = image_feature.cpu().numpy()
        # image_feature = image_feature
        range_decoder = self.decoder([img_feature, range_feature_16])
        
        points_with_features = []
        for i in range(batch_size):
            lidar = self.range_to_lidar_gpu(range_decoder[i].squeeze(0), points[i], position[i])
            points_with_features.append(self.points_rotation(lidar, np.arcsin(-1.0 * img_metas[i]['pcd_rotation'][1][0])))

        points_cat = torch.stack(points_with_features)
        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        batch_size = len(points)
        range_H16 = []
        range_H32 = []
        range_H64 = []
        position = []
        crop_image = []
        # print('This is sample_test!')
        # print(img_metas)
        for i in range(batch_size):
            # print(-1.0 * img_metas[i]['pcd_rotation'][1][0])
            # print(points[i])
            # print(points[i].shape)
            points[i] = self.points_rotation(points[i], np.arcsin(img_metas[i]['pcd_rotation'][1][0]))
            H16, H32, H64, p = self.lidar_to_range_gpu(points[i])
            range_H16.append(H16.unsqueeze(0))
            range_H32.append(H32.unsqueeze(0))
            range_H64.append(H64.unsqueeze(0))
            position.append(p)

        range_H16 = torch.cat(range_H16, dim=0)
        range_H32 = torch.cat(range_H32, dim=0)
        range_H64 = torch.cat(range_H64, dim=0)
        crop_image = F.interpolate(img[:, :, 100:, :], (64, 512), mode='bilinear')
        # print(crop_image)
        crop_image = add_noise(crop_image)
        
        range_feature_16, range_feature_32, range_feature_64 = self.stairnet([range_H16, range_H32, range_H64])
        img_feature = self.img_encoder(crop_image)
        range_decoder = self.decoder([img_feature, range_feature_16])
        points_with_features = []
        for i in range(batch_size):
            lidar = self.range_to_lidar_gpu(range_decoder[i].squeeze(0), points[i], position[i])
            points_with_features.append(self.points_rotation(lidar, np.arcsin(-1.0 * img_metas[i]['pcd_rotation'][1][0])))
            
        points_cat = torch.stack(points_with_features)
        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test with augmentation."""
        batch_size = len(points)
        range_H16 = []
        range_H32 = []
        range_H64 = []
        position = []
        crop_image = []
        
        # print('This is aug_test!')
        # print("img_metas:{}".format(img_metas))
        for i in range(batch_size):
            # print(-1.0 * img_metas[i]['pcd_rotation'][1][0])
            # print(points[i])
            # print(points[i].shape)
            points[i] = self.points_rotation(points[i], np.arcsin(img_metas[i]['pcd_rotation'][1][0]))
            H16, H32, H64, p = self.lidar_to_range_gpu(points[i])
            range_H16.append(H16.unsqueeze(0))
            range_H32.append(H32.unsqueeze(0))
            range_H64.append(H64.unsqueeze(0))
            position.append(p)

        range_H16 = torch.cat(range_H16, dim=0)
        range_H32 = torch.cat(range_H32, dim=0)
        range_H64 = torch.cat(range_H64, dim=0)
        crop_image = F.interpolate(img[:, :, 100:, :], (64, 512), mode='bilinear')
        
        crop_image = add_noise(crop_image)
        
        range_feature_16, range_feature_32, range_feature_64 = self.stairnet([range_H16, range_H32, range_H64])
        img_feature = self.img_encoder(crop_image)
        range_decoder = self.decoder([img_feature, range_feature_16])
        points_with_features = []
        for i in range(batch_size):
            lidar = self.range_to_lidar_gpu(range_decoder[i].squeeze(0), points[i], position[i])
            points_with_features.append(self.points_rotation(lidar, np.arcsin(-1.0 * img_metas[i]['pcd_rotation'][1][0])))

        points_cat = [torch.stack(pts) for pts in points_with_features]
        feats = self.extract_feats(points_cat, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(
                pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

    def points_rotation(self, pointcloud, angle):
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        scan_x = pointcloud[:, 0]
        scan_y = pointcloud[:, 1]
        x = scan_x * rot_cos - scan_y * rot_sin
        y = scan_x * rot_sin + scan_y * rot_cos
        pointcloud[:, 0] = x
        pointcloud[:, 1] = y
        return pointcloud
    
    def lidar_to_range_gpu(self, points):
        """  convert2range: points size is [N, 4]"""
        device = points.device
        pi = torch.tensor(np.pi).to(device)
        pointcloud = points[:, 0:3]
        remissions = points[:, 3]
        proj_fov_up = 3.0
        proj_fov_down = -15.0
        fov_up = proj_fov_up / 180.0 * pi
        fov_down = proj_fov_down / 180.0 * pi
        fov = abs(fov_up) + abs(fov_down)

        depth = torch.norm(pointcloud, 2, dim=1)

        scan_x = pointcloud[:, 0]
        scan_y = pointcloud[:, 1]
        scan_z = pointcloud[:, 2]

        yaw = - torch.atan2(scan_y, scan_x)
        pitch = torch.asin(scan_z / depth)

        proj_x = 0.5 * (4.55 * yaw / pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov

        # calc H16
        H = 16
        W = 128
        px = proj_x * W
        py = proj_y * H

        px = torch.floor(px)
        px = torch.min((W - 1) * torch.ones_like(px), px)
        px = torch.max(torch.zeros_like(px), px)
        px = px.long()

        py = torch.floor(py)
        py = torch.min((H - 1) * torch.ones_like(py), py)
        py = torch.max(torch.zeros_like(py), py)
        py = py.long()
        H16 = torch.full((5, H, W), 0, dtype=torch.float32).to(device)
        H16[0][py, px] = depth
        H16[1][py, px] = scan_x
        H16[2][py, px] = scan_y
        H16[3][py, px] = scan_z
        H16[4][py, px] = remissions

        # calc H32
        H = 32
        W = 256
        px = proj_x * W
        py = proj_y * H

        px = torch.floor(px)
        px = torch.min((W - 1) * torch.ones_like(px), px)
        px = torch.max(torch.zeros_like(px), px)
        px = px.long()

        py = torch.floor(py)
        py = torch.min((H - 1) * torch.ones_like(py), py)
        py = torch.max(torch.zeros_like(py), py)
        py = py.long()

        H32 = torch.full((5, H, W), 0, dtype=torch.float32).to(device)
        H32[0][py, px] = depth
        H32[1][py, px] = scan_x
        H32[2][py, px] = scan_y
        H32[3][py, px] = scan_z
        H32[4][py, px] = remissions

        #  calc H64
        H = 64
        W = 512
        px = proj_x * W
        py = proj_y * H

        px = torch.floor(px)
        px = torch.min((W - 1) * torch.ones_like(px), px)
        px = torch.max(torch.zeros_like(px), px)
        px = px.long()

        py = torch.floor(py)
        py = torch.min((H - 1) * torch.ones_like(py), py)
        py = torch.max(torch.zeros_like(py), py)
        py = py.long()
        H64 = torch.full((5, H, W), 0, dtype=torch.float32).to(device)
        H64[0][py, px] = depth
        H64[1][py, px] = scan_x
        H64[2][py, px] = scan_y
        H64[3][py, px] = scan_z
        H64[4][py, px] = remissions

        return H16, H32, H64, torch.stack([py, px], dim=1)

    def range_to_lidar_gpu(self, range_img, points, position):
        """  range2points: points size is [N, 4+12]"""
        device = range_img.device
        N = len(points)
        lidar = torch.zeros((N, 12), dtype=torch.float32).to(device)
        # lidar = torch.zeros((N, 4), dtype=torch.float32).to(device)
        lidar[:, 0:4] = points.clone().detach()
        lidar[:, 4:] = range_img[:, position[:, 0], position[:, 1]].permute((1, 0))
        return lidar


class DRB(nn.Module):
    """ ****** Dilated Residual Block ****** """

    def __init__(self, in_channels, mid_channels):
        super(DRB, self).__init__()
        " 1. Conv with Different Receptive Field"
        # D-(1,1)
        self.conv_d11 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3, 3),
                                  stride=1, padding=(1, 1), dilation=(1, 1), padding_mode='replicate')
        self.conv_d11_norm = nn.BatchNorm2d(mid_channels)
        self.conv_d11_relu = nn.ReLU(inplace=True)
        # D-(1,2)
        self.conv_d12 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(3, 3),
                                  stride=1, padding=(1, 2), dilation=(1, 2), padding_mode='replicate')
        self.conv_d12_norm = nn.BatchNorm2d(mid_channels)
        self.conv_d12_relu = nn.ReLU(inplace=True)
        # D-(1,4)
        self.conv_d14 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(3, 3),
                                  stride=1, padding=(1, 4), dilation=(1, 4), padding_mode='replicate')
        self.conv_d14_norm = nn.BatchNorm2d(mid_channels)
        self.conv_d14_relu = nn.ReLU(inplace=True)
        " 2. Channel Aggregation"
        " 3. Channel Adjustment"
        self.conv_ca = nn.Conv2d(in_channels=3 * mid_channels, out_channels=in_channels, kernel_size=(1, 1))
        self.conv_ca_norm = nn.BatchNorm2d(in_channels)
        self.conv_ca_relu = nn.ReLU(inplace=True)
        " 4. Add Operation (Residual Block)"

    def forward(self, inputs):
        # D11
        d11 = self.conv_d11(inputs)
        d11 = self.conv_d11_norm(d11)
        d11 = self.conv_d11_relu(d11)
        # D12
        d12 = self.conv_d12(d11)
        d12 = self.conv_d12_norm(d12)
        d12 = self.conv_d12_relu(d12)
        # D14
        d14 = self.conv_d14(d12)
        d14 = self.conv_d14_norm(d14)
        d14 = self.conv_d14_relu(d14)
        # concat
        d = torch.cat((d11, d12, d14), dim=1)
        d = self.conv_ca(d)
        d = self.conv_ca_norm(d)
        d = d + inputs
        output = self.conv_ca_relu(d)

        return output


class BasicBlock(nn.Module):
    """ ****** Basic Block: Init Inputs ****** """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=1):
        super(BasicBlock, self).__init__()
        self.conv_init = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=(1, 1), padding_mode="replicate")
        self.conv_init_norm = nn.BatchNorm2d(out_channels)
        self.conv_init_relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.conv_init(inputs)
        x = self.conv_init_norm(x)
        x = self.conv_init_relu(x)
        return x


class DownSampling(nn.Module):
    """ ****** Down Sampling ****** """

    def __init__(self, channels, kernel_size=(2, 2), pool='max'):  # pool = 'max' or 'avg'
        super(DownSampling, self).__init__()
        if pool == 'max':
            self.pool = nn.MaxPool2d(kernel_size)
        else:
            self.pool = nn.AvgPool2d(kernel_size)
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3),
                              stride=1, padding=(1, 1), dilation=(1, 1), padding_mode="replicate")
        self.conv_norm = nn.BatchNorm2d(channels)
        self.conv_relu = nn.ReLU()

    def forward(self, inputs):
        x = self.pool(inputs)
        x = self.conv(x)
        x = self.conv_norm(x)
        x = self.conv_relu(x)
        return x
    
    
class CAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.q_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                                   stride=2, padding=(1, 1), dilation=(1, 1), padding_mode="replicate")
        self.k_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                                   stride=2, padding=(1, 1), dilation=(1, 1), padding_mode="replicate")
        self.v_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.map = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        self.map_relu = nn.ReLU(True)

    def forward(self, inputs):
        ''' inputs:[range_img, rgb_img](features)'''
        range_img = inputs[0]
        img = inputs[1]
        batch_size, c, h, w = img.size()
        query_img = self.q_conv(range_img).view(batch_size, -1, h * w // 4)  # c x hw
        key_range = 0 * self.k_conv(img).view(batch_size, -1, h * w // 4).permute(0, 2, 1)  # hw x c
        value_range = 0 * self.v_conv(img).view(batch_size, -1, h * w)  # c x hw
        correlation = torch.bmm(query_img, key_range)  # c x c
        attention = self.softmax(correlation)  # c x c
        x = torch.bmm(attention, value_range)  # c x hw
        x = x.view(batch_size, self.out_channels, h, w) # c x h x w
        x = self.map(x)
        outputs = self.map_relu(x + img)
        return outputs

class SAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.q_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.map = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        self.map_relu = nn.ReLU(True)
        
    def forward(self, inputs):
        ''' inputs:[range_img, rgb_img](features)'''
        range_img = inputs[0]
        img = inputs[1]
        batch_size, c, h, w = img.size()
        query_img = self.q_conv(range_img).view(batch_size, -1, h * w).permute(0, 2, 1)  # hw * c
        key_range =0 * self.k_conv(img).view(batch_size, -1, h * w)  # c * hw
        value_range = 0 * self.v_conv(img).view(batch_size, -1, h * w)  # c * hw
        correlation = torch.bmm(query_img, key_range)  # hw x hw
        attention = self.softmax(correlation)  # hw x hw
        x = torch.bmm(value_range, attention)  # c x hw
        x = x.view(batch_size, self.out_channels, h, w) # c x h x w
        x = self.map(x)
        outputs = self.map_relu(x + img)
        return outputs

class Fusion(nn.Module):
    def __init__(self, in_channels):
        super(Fusion, self).__init__()
        self.SAM = SAM(in_channels, in_channels // 4)
        self.CAM = CAM(in_channels, in_channels)
        self.conv =  nn.Conv2d(in_channels=3*in_channels, out_channels=in_channels, kernel_size=(1, 1),
                                  stride=1, padding=(0, 0), dilation=(1, 1), padding_mode="replicate")
    
    def forward(self, inputs):
        range_img = inputs[0]
        img = inputs[1]
        sam = self.SAM(inputs)
        cam = self.CAM(inputs)
        x = torch.cat((sam, range_img, cam), dim=1)
        outputs = self.conv(x)
        return outputs
'''
img = torch.randn((6, 256, 16, 128))
range_img = torch.randn(6, 256, 16, 128)
model = Fusion(256, 128)
model([range_img, img]).shape

'''

class StairNet(nn.Module):
    """ ****** StairNet: Range Encoder ****** """

    def __init__(self, in_channels):
        super(StairNet, self).__init__()
        # three different size range image as input
        """1-1 Input Size: 16 x 128"""
        self.basicblock_1_1 = BasicBlock(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=2, 
                                         padding=(1, 1))
        """1-2 Input Size: 32 x 256"""
        self.basicblock_1_2 = BasicBlock(in_channels=in_channels, out_channels=64, kernel_size=(5, 5), stride=2,
                                         padding=(2, 2))
        self.DRB_1_2_1 = DRB(in_channels=64, mid_channels=64)
        self.Downsampling_1_2_1 = DownSampling(channels=64, kernel_size=(2, 2))
        self.DRB_1_2_2 = DRB(in_channels=128, mid_channels=128)

        """1-3 Input Size: 64 x 512"""
        self.basicblock_1_3 = BasicBlock(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=2,
                                         padding=(3, 3))
        self.DRB_1_3_1 = DRB(in_channels=64, mid_channels=64)
        self.Downsampling_1_3_1 = DownSampling(channels=64, kernel_size=(2, 2))
        self.DRB_1_3_2 = DRB(in_channels=128, mid_channels=128)
        self.Downsampling_1_3_2 = DownSampling(channels=128, kernel_size=(2, 2))
        self.DRB_1_3_3 = DRB(in_channels=256, mid_channels=128)

    def forward(self, inputs):
        """ inputs:[[b,c,16,128], [b,c,32,256], [b,c,64,512]] """
        H16 = inputs[0]
        H32 = inputs[1]
        H64 = inputs[2]

        d11_o1 = self.basicblock_1_1(H16)

        d12_o1 = self.basicblock_1_2(H32)
        d12 = self.DRB_1_2_1(d12_o1)
        d12 = self.Downsampling_1_2_1(d12)
        d12 = torch.cat((d11_o1, d12), dim=1)
        d12_o2 = self.DRB_1_2_2(d12)
        
        d13 = self.basicblock_1_3(H64)
        d13_o1 = self.DRB_1_3_1(d13)
        d13 = self.Downsampling_1_3_1(d13_o1)
        d13 = torch.cat((d12_o1, d13), dim=1)
        d13_o2 = self.DRB_1_3_2(d13)
        d13 = self.Downsampling_1_3_2(d13_o2)
        d13 = torch.cat((d12_o2, d13), dim=1)
        d13_o3 = self.DRB_1_3_3(d13)
        return d13_o3, d13_o2, d13_o1  # [b, 256, 8, 64], [b, 128, 16, 128], [b, 64, 32, 256]

'''
a1 = torch.zeros([2, 2, 16, 128])
a2 = torch.zeros([2, 2, 32, 256])
a3 = torch.zeros([2, 2, 64, 512])
model = StairNet(2)
b, c = model([a1, a2, a3])
'''

class Img_Encoder(nn.Module):
    def __init__(self, ):
        super(Img_Encoder, self).__init__()
        # input: resize: b 3 64 512
        self.conv_init = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7),
                                   stride=2, padding=(3, 3), dilation=(1, 1), padding_mode="replicate")
        self.conv_init_norm = nn.BatchNorm2d(64)
        self.conv_init_relu = nn.ReLU(True)
        # b 64 64 512
        self.DRB10 = DRB(in_channels=64, mid_channels=64)
        self.DRB11 = DRB(in_channels=64, mid_channels=64)

        self.DRB1_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5),
                                   stride=2, padding=(2, 2), dilation=(1, 1), padding_mode="replicate")
        self.DRB1_conv_norm = nn.BatchNorm2d(128)
        self.DRB1_conv_relu = nn.ReLU(True)
        # b 128 32 256
        self.DRB20 = DRB(in_channels=128, mid_channels=128)
        self.DRB21 = DRB(in_channels=128, mid_channels=128)
        self.DRB2_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                   stride=2, padding=(1, 1), dilation=(1, 1), padding_mode="replicate")
        self.DRB2_conv_norm = nn.BatchNorm2d(256)
        self.DRB2_conv_relu = nn.ReLU(True)

        self.DRB3 = DRB(in_channels=256, mid_channels=128)

        # b 256 16 128

    def forward(self, inputs):
        x = self.conv_init(inputs)
        x = self.conv_init_norm(x)
        x = self.conv_init_relu(x)
        x = self.DRB10(x)
        x = self.DRB11(x)
        x = self.DRB1_conv(x)
        x = self.DRB1_conv_norm(x)
        x = self.DRB1_conv_relu(x)
        x = self.DRB20(x)
        x = self.DRB21(x)
        x = self.DRB2_conv(x)
        x = self.DRB2_conv_norm(x)
        x = self.DRB2_conv_relu(x)
        x = self.DRB3(x)
        return x



class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1),
                 dilation=(1, 1), mode='bilinear', padding_mode='replicate'):
        super(UpSampling, self).__init__()
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode=mode, align_corners=False)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=padding, dilation=dilation, padding_mode=padding_mode)
        self.conv_norm = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        x = self.upsample(inputs)
        x = self.conv(x)
        outputs = self.conv_norm(x)
        return outputs


'''
inputs = torch.randn((4, 256, 16, 128))
model = UpSampling(256, 128)
model(inputs).shape
'''


class Decoder(nn.Module):
    """****** Decoder: After Fusion ******"""

    def __init__(self, ):
        super(Decoder, self).__init__()
        # input [b,c,h,w]: [b, 256, 8, 64]
        self.fusion_1 = Fusion(256)
        self.i_upsample_1 = UpSampling(256, 128)
        self.i_upsample_1_relu = nn.ReLU(True)
        self.i_DRB_1 = DRB(in_channels=128, mid_channels=128)
        self.upsample_1 = UpSampling(256, 128)
        self.upsample_1_relu = nn.ReLU(True)
        self.DRB_1 = DRB(in_channels=128, mid_channels=128)
        
        self.fusion_2 = Fusion(128)
        self.i_upsample_2 = UpSampling(128, 64)
        self.i_upsample_2_relu = nn.ReLU(True)
        self.i_DRB_2 = DRB(in_channels=64, mid_channels=64)
        self.upsample_2 = UpSampling(128, 64)
        self.upsample_2_relu = nn.ReLU(True)
        self.DRB_2 = DRB(in_channels=64, mid_channels=64)
        
        self.fusion_3 = Fusion(64)
        self.upsample_3 = UpSampling(64, 64)
        self.upsample_3_relu = nn.ReLU(True)
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(7, 7),
                                  stride=1, padding=(3, 3), dilation=(1, 1), padding_mode='replicate')

    def forward(self, inputs):
        # inputs includes:[img, range_img]
        img = inputs[0]
        range_img = inputs[1]
        
        x = self.fusion_1([range_img, img]) 
        x = self.upsample_1(x)
        x = self.upsample_1_relu(x)
        x = self.DRB_1(x)
        y = self.i_upsample_1(img)
        y = self.i_upsample_1_relu(y)
        y = self.i_DRB_1(y)
        
        x = self.fusion_2([x, y]) 
        x = self.upsample_2(x)
        x = self.upsample_2_relu(x)
        x = self.DRB_2(x)
        y = self.i_upsample_2(y)
        y = self.i_upsample_2_relu(y)
        y = self.i_DRB_2(y)
        
        x = self.fusion_3([x, y]) 
        x = self.upsample_3(x)
        x = self.upsample_3_relu(x)
        outputs = self.conv_out(x)
        
        return outputs

def add_noise(img, mean=0, std=2.0):
    device = img.device
    for i in range(img.shape[0]):
        img[i] = img[i] + (std * torch.randn(img[i].shape) + mean).to(device)
    return img
    