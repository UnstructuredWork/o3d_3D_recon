import os
import mmcv
import torch
import numpy as np

from mmdet.registry import VISUALIZERS
from mmdet.apis import inference_detector, init_detector


class Panoptic:
    def __init__(self,
                 config: str,
                 checkpoint: str,
                 device: str = 'cuda:0',
                 threshold: float = 0.8):

        self.device = torch.device(device)

        self.model_cfg = config
        self.model_chk = self.load_model(checkpoint, './panoptic/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.pth')

        self.model = init_detector(self.model_cfg, self.model_chk, device=self.device)

        self.vis = VISUALIZERS.build(self.model.cfg.visualizer)
        self.vis.dataset_meta = self.model.dataset_meta

        self.score_thr = threshold

    def load_model(self, url, file_name):
        if not os.path.exists(file_name):
            torch.hub.download_url_to_file(url, file_name)
        else:
            pass

        return file_name

    def get_panoptic(self, img):
        if img is not None:
            result = inference_detector(self.model, img)

            img = mmcv.imconvert(img, 'bgr', 'rgb')
            self.vis.add_datasample(
                name='result',
                image=img,
                data_sample=result,
                draw_gt=False,
                pred_score_thr=self.score_thr,
                show=False)

            img = self.vis.get_image()
            img = mmcv.imconvert(img, 'bgr', 'rgb')

            return img
        else:
            return np.ones((100, 100, 3), dtype=np.uint8)