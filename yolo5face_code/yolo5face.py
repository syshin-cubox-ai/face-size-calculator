import os
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch

from backbone_code.yolo5face_code.utils import check_img_size, letterbox, non_max_suppression_face
from backbone_code.yolo5face_code.yolo import Model


class YOLO5Face:
    def __init__(
        self,
        model_cfg: str,
        model_weights: str,
        img_size: int,
        conf_thres: float,
        iou_thres: float,
        device: str,
    ):
        """
        Args:
            model_cfg: YOLO5Face's config file path.
            model_weights: YOLO5Face's weights file path.
            img_size: Inference image size.
            conf_thres: Confidence threshold.
            iou_thres: IoU threshold.
            device: Device to inference.
        """
        assert os.path.exists(model_cfg), f'model_cfg is not exists: {model_cfg}'
        assert os.path.exists(model_weights), f'model_weights is not exists: {model_weights}'
        assert 0 <= conf_thres <= 1, 'conf_thres must be between 0 and 1.'
        assert 0 <= iou_thres <= 1, 'iou_thres must be between 0 and 1.'
        assert device in ['cpu', 'cuda'], f'device is invalid: {device}'

        self.model_cfg = model_cfg
        self.model_weights = model_weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device(device)

        # Load model
        model = Model(self.model_cfg).to(self.device)
        model.load_state_dict(torch.load(self.model_weights))
        model.float().fuse().eval()
        self.model = model

    def transform_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Resizes the input image to fit img_size while maintaining aspect ratio.
        It also converts ndarray to tensor.
        (BGR to RGB, HWC to CHW, 0~1 normalization, and adding batch dimension)
        """

        h, w = img.shape[:2]
        r = self.img_size / max(h, w)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=interp)

        img_size = check_img_size(self.img_size, self.model.stride.max())
        img = letterbox(img, new_shape=img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, HWC to CHW

        img = torch.as_tensor(img, dtype=torch.float32)
        img /= 255.0  # 0~255 to 0.0~1.0
        img = img.unsqueeze(0)
        return img

    def scale_coords(self, img1_shape, coords: torch.Tensor, img0_shape, ratio_pad=None) -> torch.Tensor:
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, 5:15:2] -= pad[0]  # x padding
        coords[:, 6:15:2] -= pad[1]  # y padding
        coords[:, :4] /= gain
        coords[:, 5:15] /= gain

        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 5:15:2].clamp_(0, img0_shape[1])  # x axis
        coords[:, 6:15:2].clamp_(0, img0_shape[0])  # y axis
        return coords

    def detect_one(self, img: np.ndarray) -> Union[torch.Tensor, None]:
        """
        Perform face detection on a single image.

        Args:
            img: Input image read using OpenCV. (HWC, BGR)
        Return:
            pred:
                Post-processed predictions. Shape=(number of faces, 16)
                16 is composed of bbox coordinates, confidence, landmarks coordinates, and class number.
                The coordinate format is x1y1x2y2 (bbox), xy per point (landmarks).
                The unit is image pixel.
                If no face is detected, output None.
        """

        # Transform image
        original_img_shape = img.shape[:2]
        img = self.transform_image(img).to(self.device)
        transformed_img_shape = img.shape[2:]

        # Inference
        with torch.no_grad():
            pred = self.model(img)[0]
            pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]

        # Rescale coordinates from inference size to input image size
        if pred.shape[0] > 0:
            pred = self.scale_coords(transformed_img_shape, pred, original_img_shape)
            return pred
        else:
            return None

    def parse_prediction(self, pred: torch.Tensor) -> Tuple[List, List, List]:
        # Parse prediction to bbox, confidence, landmarks.
        bbox = pred[:, :4].round().to(torch.int32).tolist()
        conf = pred[:, 4].tolist()
        landmarks = pred[:, 5:15].round().to(torch.int32).tolist()
        # class_num = pred[:, 15].to(torch.int32).tolist()
        return bbox, conf, landmarks
