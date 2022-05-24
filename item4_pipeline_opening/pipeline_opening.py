from PIL import Image
import cv2
import json

from item4_pipeline_opening.yolo import YOLO


class PipelineOpeningDetection(object):
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
    res_dir = "/home/zranguai/Project/SDK-learn/item4_pipeline_opening/model_data"
    detector = PipelineOpeningDetection(res_dir)

    img = cv2.imread('/home/zranguai/Project/sdk_v0.0.1/test_data/item4_AB3_209.jpg')
    imgs = list()
    imgs.append(img)
    result = detector.detect(imgs)
    print(result)
