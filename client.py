# -*- coding:utf-8 -*-
# Author: zranguai
# Time: 2022-5-17

import argparse
import socket
import json
from loguru import logger  # 日志文件
import sys
import cv2

import tcp_tools


def parser_args():
    parser = argparse.ArgumentParser("Tcp client parser")
    parser.add_argument("port", type=int, help="server port")
    parser.add_argument("image", help="input type, eg. image, video and webcam")
    parser.add_argument("--ip", default="127.0.0.1", help="tcp server ip")
    parser.add_argument("--item_ids", type=list, default=[4, 6], help="需要开启服务的id，4: 管道是否封闭")
    args = parser.parse_args()
    return args


def client_socker(ip, port):
    """
    客户端开启服务
    :param ip:
    :param port:
    :return: socket
    """
    client_sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sk.connect((ip, port))
    return client_sk


def process_msg(rsp, img_path):
    """
    处理接受过来的消息
    :param rsp:
    :return:
    """
    rsp = json.loads(rsp.decode("utf-8"))  # bytearray->json
    # 判断传送是否成功
    if rsp["status"] != 0:
        logger.info("ERROR: {}".format(rsp["msg"]))
        sys.exit(1)
    # 判断是否有危险发生
    if 1 == rsp["exception"]:
        logger.info("Exception appears!!!")
    else:
        print("Normal state")
    # 读取图片进行绘画出来
    img = cv2.imread(img_path)
    # 如果有检测结果进行画框，否则直接显示图片
    if len(rsp["details"]) > 0:
        for item_result in rsp["details"]:
            logger.info("Item{}:{}".format(item_result["item_id"], item_result["item_desc"]))
            # 出现异常
            if 1 == item_result["exception"]:
                logger.info("One exception here!")
                cv2.putText(  # 画出有危险标识
                    img,  # img
                    "Exception",  # text
                    (30, 70),  # location
                    cv2.FONT_HERSHEY_SIMPLEX,  # font
                    2,  # fontscale
                    (0, 0, 255),  # color
                    3,  # thickness
                )

            for obj in item_result['objects']:
                color = (0, 0, 255) if obj['risk_level'] >= 2 else (0, 255, 0)
                thickness = 8 if obj['risk_level'] >= 2 else 2
                if obj['type'] == 'detection':
                    pos = obj['position']
                    cv2.rectangle(img, (pos['x1'], pos['y1']), (pos['x2'], pos['y2']), color, thickness)
                elif obj['type'] == 'segmentation':
                    pos = obj['position']
                    for pt in pos:
                        cv2.circle(img, (pt['x'], pt['y']), 2, (0, 255, 255), 2)

    new_w = 1024
    img = cv2.resize(img, (new_w, int(img.shape[0] / img.shape[1] * new_w)))  # 图片， 宽/高
    cv2.imshow("process_img", img)

    k = cv2.waitKey(0)
    if k == 27 or k == ord("q") or k == ord("Q"):
        cv2.destroyAllWindows()


def client_main(args):
    client = client_socker(args.ip, args.port)  # 客户端连接服务端开启服务

    req = dict()
    req["img_path"] = args.image
    req["item_ids"] = args.item_ids
    req = json.dumps(req).encode("utf-8")  # encode转换成binary格式方便传输
    tcp_tools.send_msg(client, req)  # 客户端发送消息

    rsp = tcp_tools.recv_msg(client, 30 * 60)  # 客户端接受消息
    process_msg(rsp, args.image)


if __name__ == '__main__':
    args = parser_args()
    client_main(args)
