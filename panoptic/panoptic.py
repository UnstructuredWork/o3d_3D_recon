import os

import cv2
import mmcv
import torch
import time
import numpy as np

# from mmdet.registry import VISUALIZERS
from mmdet.apis import inference_detector, init_detector

# PANOPTIC_PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
#          (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
#          (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
#          (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
#          (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
#          (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
#          (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
#          (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
#          (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
#          (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
#          (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
#          (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
#          (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
#          (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
#          (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
#          (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
#          (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
#          (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
#          (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
#          (246, 0, 122), (191, 162, 208), (255, 255, 128), (147, 211, 203),
#          (150, 100, 100), (168, 171, 172), (146, 112, 198), (210, 170, 100),
#          (92, 136, 89), (218, 88, 184), (241, 129, 0), (217, 17, 255),
#          (124, 74, 181), (70, 70, 70), (255, 228, 255), (154, 208, 0),
#          (193, 0, 92), (76, 91, 113), (255, 180, 195), (106, 154, 176),
#          (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
#          (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
#          (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
#          (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
#          (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
#          (146, 139, 141), (70, 130, 180), (134, 199, 156), (209, 226, 140),
#          (96, 36, 108), (96, 96, 96), (64, 170, 64), (152, 251, 152),
#          (208, 229, 228), (206, 186, 171), (152, 161, 64), (116, 112, 0),
#          (0, 114, 143), (102, 102, 156), (250, 141, 255)]
PANOPTIC_PALETTE = [(0, 0, 0), (127, 141, 255), (119, 11, 32), (0, 50, 142), (220, 20, 60), (163, 255, 0),
                    (67, 26, 255), (0, 0, 230), (104, 84, 109), (45, 89, 255), (210, 170, 100), (209, 99, 106)]
NUM_CLASSES = len(PANOPTIC_PALETTE)

class Panoptic:
    def __init__(self,
                 config: str,
                 checkpoint: str,
                 classes: list = None,
                 device: str = 'cuda:0',
                 threshold: float = 0.8):

        self.device = torch.device(device)

        self.model_cfg = config
        # self.model_chk = self.load_model(checkpoint, './panoptic/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.pth')
        self.model_chk = checkpoint
        self.model = init_detector(self.model_cfg, self.model_chk, device=self.device)

        # self.vis = VISUALIZERS.build(self.model.cfg.visualizer)
        # self.vis.dataset_meta = self.model.dataset_meta

        self.score_thr = threshold
        self.classes = classes

    def load_model(self, url, file_name):
        if not os.path.exists(file_name):
            torch.hub.download_url_to_file(url, file_name)
        else:
            pass

        return file_name

    def get_panoptic(self, img):
        t1 = time.time()
        img = cv2.resize(img, dsize=[666, 400])
        result = inference_detector(self.model, img)

        result_sem_seg = result.pred_panoptic_seg.sem_seg.cpu()
        result_sem_seg = np.squeeze(result_sem_seg)

        # print(len(np.where(result_sem_seg > 133)))
        # print(result_sem_seg)
        # print(result_sem_seg.shape)
        labels = result.pred_instances.labels.cpu()
        masks  = result.pred_instances.masks.cpu()
        # print(labels)
        t2 = time.time()
        #
        mask = np.ones_like(img, dtype=np.uint8) * 255

        for cls_id in np.unique(result_sem_seg):
            if cls_id >= NUM_CLASSES:
                continue
            mask[result_sem_seg == cls_id] = PANOPTIC_PALETTE[cls_id]

        for id, cls in enumerate(labels):
            mask[masks[id, :, :]] = PANOPTIC_PALETTE[cls]

        t3 = time.time()

        # cv2.imshow('', mask)
        # cv2.waitKey(1)


        # print(mask)
        print("t1: ", (t2 - t1) * 1000)
        print("t2: ", (t3 - t1) * 1000)
        mask = cv2.resize(mask, dsize=[2048, 1536])
        return mask

