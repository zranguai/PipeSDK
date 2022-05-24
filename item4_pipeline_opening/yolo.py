import os
import numpy as np
import colorsys
import torch
import torch.nn as nn
from PIL import ImageFont

from item4_pipeline_opening.nets.yolo4 import YoloBody
from item4_pipeline_opening.utils import DecodeBox, letterbox_image, non_max_suppression, yolo_correct_boxes


class YOLO(object):

    def __init__(self, res_dir):
        self.load_weight = [os.path.join(res_dir, "model.pth")]
        self.anchors_path = os.path.join(res_dir, "yolo_anchors.txt")
        self.classes_path = os.path.join(res_dir, "voc_hole.txt")
        self.font_file = os.path.join(res_dir, 'simhei.ttf')
        self.model_image_size = (416, 416, 3)
        self.confidence = 0.5
        self.cuda = True if torch.cuda.is_available() else False

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()  # 网络的初始化以及decode的初始化

    def _get_class(self):
        """
        获得所有的分类名字
        :return: class_name
        """
        classes_path = self.classes_path
        with open(classes_path, "r") as f:
            class_name = f.readlines()
        class_names = [c.strip() for c in class_name]
        return class_names

    def _get_anchors(self):
        """
        获得所有的先验框
        :return:
        """
        anchors_path = self.anchors_path
        with open(anchors_path, "r") as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    def generate(self):
        """
        网络的初始化，以及decode的初始化
        :return: None
        """
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

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self, image):
        """
        检测图片
        :param image:
        :return:
        """
        image_shape = np.array(np.shape(image)[0:2])  # image_shape: array（[1080,1920]

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))  # 裁剪图片
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0  # 归一化
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)

        images = []
        images.append(photo)
        images = np.asarray(images)  # 给图片增加batch_size维度 shape=(1, 3, 416, 416)

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
            output,  # shape=(1, 10647, 7)
            len(self.class_names),  # 2
            conf_thres=self.confidence,  # 0.5
            nms_thres=0.3
        )
        try:
            batch_detections = batch_detections[0].cpu().numpy()  # shape:(2, 7)
        except:
            return []

        top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
        top_conf = batch_detections[:, 4] * batch_detections[:, 5]  # 置信度 * 类别分数
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])

        top_xmin = np.expand_dims(top_bboxes[:, 0], -1)  # 把x_min合在一起
        top_ymin = np.expand_dims(top_bboxes[:, 1], -1)
        top_xmax = np.expand_dims(top_bboxes[:, 2], -1)
        top_ymax = np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条(转换成图片的尺度)
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.model_image_size[0], self.model_image_size[1]]),
                                   image_shape)

        # font = ImageFont.truetype(font=self.font_file, size=np.floor(3e-2 * np.shape(image)[1] - 30).astype('int32'))
        # thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

        # 将预测的结果传出去
        count_num = []
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框
            label = '{} {:.2f}'.format(predicted_class, score)
            data = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'score': 0, 'is_open': 0}

            data['x1'] = float(left)
            data['y1'] = float(top)
            data['x2'] = float(right)
            data['y2'] = float(bottom)

            if label.split(' ')[0] == 'no':
                data['is_open'] = 1
            data['score'] = round(float(score), 4)
            count_num.append(data)
        return count_num
