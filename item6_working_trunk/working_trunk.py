from PIL import Image
import cv2
import json

from item6_working_trunk.yolo import YOLO


class WorkingTrunk(object):
    # 模型数据相关的目录
    def __init__(self, res_dir):
        self.yolov4 = YOLO(res_dir)

    def detect(self, imgs):
        cv_bgr = imgs[0]
        pil_img = Image.fromarray(cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB))  # 将BGR图像转换成RGB图像
        labels = self.yolov4.detect_image(pil_img)
        result = []
        result.append(labels)
        return json.dumps(result)


if __name__ == '__main__':
    res_dir = "/home/zranguai/Project/Pipeline/SDK-learn/item6_working_trunk/model_data"
    detector = WorkingTrunk(res_dir)

    img_path = "/home/zranguai/Project/Pipeline/SDK-learn/test_data/item6_2.jpg"
    img = cv2.imread(img_path)
    result = detector.detect([img])
    print(result)
