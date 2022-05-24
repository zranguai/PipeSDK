# -*- coding:utf-8 -*-
# Author: zranguai
# Time: 2022-5-17

import argparse
import json
import socket
import traceback

import cv2
from loguru import logger

import tcp_tools


def parser_args():
    parser = argparse.ArgumentParser("Tcp server parser")
    parser.add_argument("ip", help="tcp server ip")
    parser.add_argument("port", type=int, help="server port")
    parser.add_argument("--max_connect", default=128, help="该服务的最大连接数")
    args = parser.parse_args()
    return args


def svr_socker(ip, port):
    svr_sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # ipv4, tcp
    svr_sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    svr_sk.bind((ip, port))
    return svr_sk


class ItemsScheduler(object):
    def __init__(self):
        from item4_pipeline_opening.pipeline_opening import PipelineOpeningDetection
        from item6_working_trunk.working_trunk import WorkingTrunk

        item4_pipeline_opening_res_dir = "item4_pipeline_opening/model_data"
        item6_pipeline_opening_res_dir = "item6_working_trunk/model_data"

        logger.info("Init item4_pipeline_opening...")
        self._item4_pipeline_opening = PipelineOpeningDetection(item4_pipeline_opening_res_dir)  # init model
        logger.info("Init item6_working_trunk...")
        self._item6_working_trunk = WorkingTrunk(item6_pipeline_opening_res_dir)  # init model

    def process(self, conn):
        req = tcp_tools.recv_msg(conn)  # req是bytearray类型
        req = json.loads(req.decode("utf-8"))  # 从客户端拿到item_ids和img信息

        logger.info("Start reading data...")

        if "img_path" in req.keys():  # 后面扩展从客户端读取的其他形式
            img = cv2.imread(req["img_path"])  # cv2读取出来的图片形式是BGR的

        rsp = dict()  # 存储总的
        rsp["status"] = 0  # 传送的状态
        rsp["msg"] = "finished"
        rsp["exception"] = 0
        rsp["details"] = list()

        logger.info("Start detecting")

        if 4 in req["item_ids"]:
            try:
                det_result = self._item4_pipeline_opening.detect([img])
                det = json.loads(det_result)
                logger.info(det)
                logger.info("- - - - - - - - - - - -")

                # 将结果打包好给客户端
                if len(det) > 0:
                    det_inner = det[0]
                    if len(det_inner) > 0:
                        out = dict()  # 存储单个的
                        out["item_id"] = 4
                        out["item_desc"] = "is pipeline opening"
                        out["exception"] = 0
                        out["objects"] = list()
                        for item in det_inner:
                            obj = dict()
                            if 1 == item["is_open"]:
                                out["exception"] = 1
                                rsp["exception"] = 1
                                obj["risk_level"] = 2  # 管道是open的风险等级为2
                            else:  # 管道关闭情况
                                obj["risk_level"] = 0
                            obj["name"] = "pipeline opening"
                            obj["type"] = "detection"
                            obj["position"] = {  # 管道坐标和分数
                                'x1': int(item['x1']),
                                'y1': int(item['y1']),
                                'x2': int(item['x2']),
                                'y2': int(item['y2']),
                                'conf': item['score'],
                            }
                            out["objects"].append(obj)  # 单个进行append
                        rsp["details"].append(out)
            except:
                traceback.print_exc()
        if 6 in req["item_ids"]:
            try:
                det_result = self._item6_working_trunk.detect([img])
                det = json.loads(det_result)
                print(det)
                logger.info(det)
                logger.info("- - - - - - - - - - - -")

                # 将结果打包好给客户端
                if len(det) > 0:
                    det = det[0]
                    if len(det) > 0:
                        out = dict()
                        out['item_id'] = 6
                        out['item_desc'] = 'if person is under working trunk'
                        out['exception'] = 0
                        out['objects'] = list()
                        for item in det:
                            obj = dict()
                            if 'risk_level' in item.keys():
                                if item['risk_level'] >= 2:
                                    out['exception'] = 1
                                    rsp['exception'] = 1
                                obj['risk_level'] = item['risk_level']
                            else:
                                obj['risk_level'] = 0
                            obj['name'] = item['type']
                            obj['type'] = 'detection'
                            obj['position'] = {
                                'x1': int(item['x1']),
                                'y1': int(item['y1']),
                                'x2': int(item['x2']),
                                'y2': int(item['y2']),
                                'conf': item['score'],
                            }
                            out['objects'].append(obj)
                        rsp['details'].append(out)
            except:
                traceback.print_exc()
        # 将打包好的结果发送给客户端
        rsp = json.dumps(rsp).encode("utf-8")  # -> json -> bytearray
        tcp_tools.send_msg(conn, rsp)


def svr_main(args):
    listen_svr_sk = svr_socker(args.ip, args.port)

    # init worker
    worker = ItemsScheduler()

    # 进行不断监听是否有服务端过来连接
    while True:
        try:
            listen_svr_sk.listen(args.max_connect)
            logger.info("Start Listening...")

            conn, client_addr = listen_svr_sk.accept()
            logger.info("conn: {}".format(conn))
            logger.info("client_addr: {}".format(client_addr))
            worker.process(conn)  # recv and send in process

            conn.close()  # 关闭当前客户端的服务
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")
        except:
            logger.info("Other error")
            listen_svr_sk.close()  # 关闭整个服务


if __name__ == '__main__':
    args = parser_args()
    svr_main(args)
