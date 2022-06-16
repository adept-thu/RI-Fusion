import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
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
class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

        # Encoder, Fusion and Decoder
        self.stairnet = StairNet(5)  # inputs: [b, 5, h, w]
        self.img_encoder = Img_Encoder()  # inputs: [b, 3, h, w]
        self.decoder = Decoder()  # inputs:[fusion, d13_o1]
        
    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img=None,
                      gt_bboxes_ignore=None):
        """Training forward function.
        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
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
        print(range_H16.shape)
        range16 = range_H16.cpu().numpy()
        range16 = range16[0,0,:,:]  
        print(range16.shape)
        imageio.imsave("test.png",range16)


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
            
        x = self.extract_feat(points_with_features, img_metas)
        # x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        batch_size = len(points)
        range_H16 = []
        range_H32 = []
        range_H64 = []
        position = []
        crop_image = []
        
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
            

        x = self.extract_feat(points_with_features, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function with augmentaiton."""
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
        crop_image = F.interpolate(img[:, :, 100:, :], (64, 512), mode='bilinear')
        
        crop_image = add_noise(crop_image)
        
        range_feature_16, range_feature_32, range_feature_64 = self.stairnet([range_H16, range_H32, range_H64])
        img_feature = self.img_encoder(crop_image)
        range_decoder = self.decoder([img_feature, range_feature_16])
        points_with_features = []
        for i in range(batch_size):
            lidar = self.range_to_lidar_gpu(range_decoder[i].squeeze(0), points[i], position[i])
            points_with_features.append(self.points_rotation(lidar, np.arcsin(-1.0 * img_metas[i]['pcd_rotation'][1][0])))
            
        x = self.extract_feat(points_with_features, img_metas)
        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
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

    
class Grid(object):
    def __init__(self, d1, d2, rotate = 1, ratio = [0.8, 0.6], mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        device = img.device
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
        
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image 
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h*h + w*w)))
        
        dh = np.random.randint(self.d1[0], self.d2[0])
        dw = np.random.randint(self.d1[1], self.d2[1])
        
        #d = self.d
        # maybe use ceil? but i guess no big difference
        self.lh = math.ceil(dh*self.ratio[0])
        self.lw = math.ceil(dw*self.ratio[1])
        mask = np.zeros((hh, hh), np.float32)
        for i in range(-1, hh//dh+1):
                s = dh*i + dh
                t = s+self.lh
                s = min(s, hh)
                t = min(t, hh)
                mask[s:t,:] = 1
        for i in range(-1, hh//dw+1):
                s = dw*i + dw
                t = s+self.lw
                s = min(s, hh)
                t = min(t, hh)
                mask[:,s:t] = 1
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

        mask = torch.from_numpy(mask).float().to(device)
        if self.mode == 1:
            mask = 1-mask
            
        # print(mask.shape, img.shape)
        mask = mask.expand_as(img)
        img = img * mask 

        return img

    
class GridMask(nn.Module):
    def __init__(self, d1, d2, rotate = 1, ratio = [0.8, 0.6], mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        if not self.training:
            return x
        n,c,h,w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))
        y = torch.cat(y).view(n,c,h,w)
        return y

def random_channels(img):
    device = img.device
    ''' List : change channels'''
    rank = [[0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]]
    batch = img.shape[0]
    for i in range(batch):
        r = random.randint(0, 2)
        if r >= 2:
            r = random.randint(0, 7)
            img[i][0:3] = img[i][r]
    return img

def add_noise(img, mean=0, std=2.0):
    device = img.device
    for i in range(img.shape[0]):
        img[i] = img[i] + (std * torch.randn(img[i].shape) + mean).to(device)
    return img
    
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    """Get the 3D rotation matrix.

    Args:
        rot_mat_T (np.ndarray): Transposed rotation matrix.
        angle (float): Rotation angle.
        axis (int): Rotation axis.
    """
    # kitti axis = 2
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    