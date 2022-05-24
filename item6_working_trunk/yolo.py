import os
import numpy as np
import torch
import torch.nn as nn

from item6_working_trunk.nets.yolo4 import YoloBody
from item6_working_trunk.utils.utils import DecodeBox, letterbox_image, non_max_suppression, yolo_correct_boxes
from item6_working_trunk.utils.working_trunk_post_process import WorkingTrunkPost


class YOLO(object):
    def __init__(self, res_dir):
        self.load_weight = [os.path.join(res_dir, "model.pth")]
        self.anchors_path = os.path.join(res_dir, "yolo_anchors.txt")
        self.classes_path = os.path.join(res_dir, "pipline_classes.txt")
        self.font_file = os.path.join(res_dir, "simhei.ttf")
        self.model_image_size = (416, 416, 3)
        self.confidence = 0.5
        self.cuda = True if torch.cuda.is_available() else False

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()  # 网络的初始化以及decode的初始化

    def _get_class(self):
        classes_path = self.classes_path
        with open(classes_path, "r") as f:
            class_name = f.readlines()
        class_names = [c.strip() for c in class_name]
        return class_names

    def _get_anchors(self):
        anchors_path = self.anchors_path
        with open(anchors_path, "r") as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(",")]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    def generate(self):
        self.net = YoloBody(len(self.anchors[0]), len(self.class_names))
        self.net = self.net.eval()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        state_dict = torch.load(self.load_weight[0], map_location=device)
        self.net.load_state_dict(state_dict)  # 加载权重

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        self.yolo_decodes = []
        for i in range(3):
            decode_result = DecodeBox(self.anchors[i], len(self.class_names), (self.model_image_size[1], self.model_image_size[0]))
            self.yolo_decodes.append(decode_result)

    def detect_image(self, image):
        """
        进行检测图片并画图
        :param image:
        :return:
        """
        image_shape = np.array(np.shape(image)[0:2])  # [1080 1920]

        # 裁剪图片到 416 * 416
        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0  # 归一化
        photo = np.transpose(photo, (2, 0, 1))  # 将channel维度提到前面来
        photo = photo.astype(np.float32)

        images = []
        images.append(photo)
        images = np.asarray(images)  # 给图片增加batch_size维度

        with torch.no_grad():
            images = torch.from_numpy(images)  # numpy转换成torch
            if self.cuda:
                images = images.cuda()
            # forward
            outputs = self.net(images)

        output_list = []
        for i in range(3):
            decode_result = self.yolo_decodes[i](outputs[i])  # 进行decode
            output_list.append(decode_result)
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(
            output,  # shape=(1, 10647)
            len(self.class_names),  # 5
            conf_thres=self.confidence,  # 0.5
            nms_thres=0.3
        )
        try:
            batch_detections = batch_detections[0].cpu().numpy()  # shape:()
        except:
            return []

        top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence  # 筛选需要的index
        top_conf = batch_detections[:, 4] * batch_detections[:, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])

        top_xmin = np.expand_dims(top_bboxes[:, 0], -1)  # 把x_min合在一起
        top_ymin = np.expand_dims(top_bboxes[:, 1], -1)
        top_xmax = np.expand_dims(top_bboxes[:, 2], -1)
        top_ymax = np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条转换成原始图片尺度
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.model_image_size[0], self.model_image_size[1]]),
                                   image_shape)

        # post process logic for working trunk
        wt_postprocess = WorkingTrunkPost(  # init wt_postprocess
            top_conf,
            top_label,
            boxes
        )
        wr_result = wt_postprocess.process_data()
        return wr_result
