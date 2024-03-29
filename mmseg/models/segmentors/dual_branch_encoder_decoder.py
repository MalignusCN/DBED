# Copyright (c) OpenMMLab. All rights reserved.
from re import I
from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class DualBranchEncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone_rgb,
                 backbone_nir,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 attention_module=None,
                 decouple_loss=None,
                 latent_convert=None):
        super(DualBranchEncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone_rgb.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone_rgb.pretrained = pretrained
            backbone_nir.pretrained = pretrained
        self.backbone_rgb = builder.build_backbone(backbone_rgb)
        self.backbone_nir = builder.build_backbone(backbone_nir)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if attention_module is not None:
            self.with_att = True
            self.attention_module = builder.build_neck(attention_module)
        else:
            self.with_att = False
            self.attention_module = None
        if latent_convert is not None:
            self.latent_convert = nn.ModuleList(
                    [nn.Conv2d(input_feature_dim, input_feature_dim, kernel_size=1, padding=1) \
                        for input_feature_dim in latent_convert['input_feature_dims']]
                )
            self.decouple_stage_list = latent_convert['decouple_stage_list']
        else:
            self.latent_convert = None
            self.decouple_stage_list = None
        if decouple_loss is not None:
            self.decouple_loss = builder.build_loss(decouple_loss)
        else:
            self.decouple_loss = None
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat_nir(self, img_nir):
        """Extract features from nir channels."""
        x = self.backbone_nir(img_nir)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat_rgb(self, img_rgb):
        """Extract features from rgb channels."""
        x = self.backbone_rgb(img_rgb)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat(self, img):
        """Extract features from rgb channels."""
        x = self.backbone_rgb(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        img_NIR = torch.unsqueeze(img[:, 0, :, :], 1)
        img_RGB = img[:, 1:, :, :]

        x_nir = self.extract_feat_nir(img_NIR)
        x_rgb = self.extract_feat_rgb(img_RGB)
        # x = x_nir + x_rgb

        if self.with_att:
            x = self.attention_module(x_nir, x_rgb)

        if self.decouple_stage_list is not None and self.decouple_loss is not None:
            new_x_nir = []
            new_x_rgb = []
            for idx in range(len(self.decouple_stage_list)):
                temp_nir = self.latent_convert[idx](x_nir[self.decouple_stage_list[idx]])
                temp_rgb = self.latent_convert[idx](x_rgb[self.decouple_stage_list[idx]])
                new_x_nir.append(temp_nir)
                new_x_rgb.append(temp_rgb)

        # x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img_NIR = torch.unsqueeze(img[:, 0, :, :], 1)
        img_RGB = img[:, 1:, :, :]


        x_nir = self.extract_feat_nir(img_NIR)
        x_rgb = self.extract_feat_rgb(img_RGB)

        if self.with_att:
            x = self.attention_module(x_nir, x_rgb)

        losses = dict()

        if self.decouple_stage_list is not None and self.decouple_loss is not None:
            new_x_nir = []
            new_x_rgb = []
            for idx in range(len(self.decouple_stage_list)):
                temp_nir = self.latent_convert[idx](x_nir[self.decouple_stage_list[idx]])
                temp_rgb = self.latent_convert[idx](x_rgb[self.decouple_stage_list[idx]])
                new_x_nir.append(temp_nir)
                new_x_rgb.append(temp_rgb)
            loss_decouple = self.decouple_loss(new_x_nir, new_x_rgb)
            losses.update({'loss_decouple': loss_decouple})

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            # remove padding area
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        if self.out_channels == 1:
            output = F.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if self.out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        if int(img_meta[0]['ori_filename'][:-4]) % 40 == 0:
            file_name = img_meta[0]['ori_filename'][:-4] + ".jpg"
            save_path = "/home/wangyuhao/vis/UPerNet_convnext_DBED"
            new_np_label = np.zeros((seg_pred[0].shape[0], seg_pred[0].shape[1], 3), dtype=np.uint16)
            for h in range(new_np_label.shape[0]):
                for w in range(new_np_label.shape[1]):
                    if seg_pred[0][h][w] == 1:
                        new_np_label[h][w] = [0, 255, 0]
                    elif seg_pred[0][h][w] == 2:
                        new_np_label[h][w] = [0, 255, 255]
                    elif seg_pred[0][h][w] == 3:
                        new_np_label[h][w] = [255, 255, 0]
                    elif seg_pred[0][h][w] == 4:
                        new_np_label[h][w] = [0, 0, 255]
                    elif seg_pred[0][h][w] == 0:
                        new_np_label[h][w] = [255, 0, 0]
                    elif seg_pred[0][h][w] == 5:
                        new_np_label[h][w] = [0, 0, 0]
            cv2.imwrite(os.path.join(save_path, file_name), cv2.cvtColor(\
                new_np_label, cv2.COLOR_RGB2BGR))
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        if self.out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred