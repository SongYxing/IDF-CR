from typing import Sequence, Dict, Union
import math
import time

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data

from utils.file import load_file_list
from utils.image import center_crop_arr, augment, random_crop_arr


class MyDataset(data.Dataset):

    def __init__(
            self,
            lq_file_list: str,
            gt_file_list: str,
            out_size: int,
            crop_type: str,
            use_hflip: bool
    ) -> "MyDataset":
        super(MyDataset, self).__init__()
        self.lq_file_list = lq_file_list
        self.gt_file_list = gt_file_list
        self.lq_paths = load_file_list(lq_file_list)
        self.gt_paths = load_file_list(gt_file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"

        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)

        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape

        # load lq image
        lq_path = self.lq_paths[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(lq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {lq_path}"

        if self.crop_type == "center":
            pil_img_lq = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_lq = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_lq = np.array(pil_img)
            assert pil_img_lq.shape[:2] == (self.out_size, self.out_size)
        img_lq = (pil_img_lq[..., ::-1] / 255.0).astype(np.float32)

        # random horizontal flip
        img_lq = augment(img_lq, hflip=self.use_hflip, rotation=False, return_status=False)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)

        return dict(jpg=target, txt="", hint=source)

    def __len__(self) -> int:
        return len(self.gt_paths)
