# -*- coding:utf-8 -*-
# Author: zranguai
# Time: 2022-5-21
import numpy as np

class WorkingTrunkPost(object):
    def __init__(self, top_conf, top_label, boxes):
        self.top_conf = top_conf
        self.top_label = top_label
        self.boxes = boxes

    def process_data(self):
        """
        1. 得到每个类别的框，放进对应类别的框里面里
        2. 将所有类别(除了person类)(框，score, type)放进pr_result里
        3. 将person类别的分数保存进person_score
        :return:
        """
        console_bbox = []  # 控制台
        crane_bbox = []  # 起重机
        hook_bbox = []  # 吊机臂
        excavator_bbox = []  # 挖机臂
        person_bbox = []

        pr_result = []
        person_score = []
        for i, c in enumerate(self.top_label):
            if c == 0:
                console_bbox.append(self.boxes[i].astype('int32'))
                self.add_print_result(self.boxes[i].astype('int32'), self.top_conf[i], pr_result, de_type='console')
            if c == 1:
                crane_bbox.append(self.boxes[i].astype('int32'))
                self.add_print_result(self.boxes[i].astype('int32'), self.top_conf[i], pr_result, de_type='console')
            if c == 2:
                hook_bbox.append(self.boxes[i].astype('int32'))
                self.add_print_result(self.boxes[i].astype('int32'), self.top_conf[i], pr_result, de_type='console')
            if c == 3:
                excavator_bbox.append(self.boxes[i].astype('int32'))
                self.add_print_result(self.boxes[i].astype('int32'), self.top_conf[i], pr_result, de_type='console')
            if c == 4:
                person_bbox.append(self.boxes[i].astype('int32'))
                person_score.append(self.top_conf[i])
        # 判断人是否在这些臂的下面
        p_bbox = []
        for _, coor in enumerate(person_bbox):
            p_bbox.append(coor.tolist())  # tolist() 将numpy类型转换成list类型
        if person_bbox == []:
            print('safe, no person')
        elif crane_bbox == [] and hook_bbox == [] and excavator_bbox == []:
            print("safe, no arm")
        else:
            raw_v_dangerous_b = []
            raw_dangerous_b = []
            raw_v_dangerous_indx = []
            raw_dangerous_indx = []

            for idx_co, coor_co in enumerate(console_bbox):  # 这里表示有控制台
                cons_bb = self.scale_expand(coor_co, 1.1)  # 这里表示将控制台放大1.1倍

                v_dangerous_b_crane, vd_cperson_idx, dangerous_b_crane, d_cperson_idx = self.loc_console_with_arm_per(idx_co, cons_bb, crane_bbox, person_bbox) #控制台box,吊车box
                v_dangerous_b_crane = self.list_is_empty(v_dangerous_b_crane)  # 判断是不是为空
                dangerous_b_crane = self.list_is_empty(dangerous_b_crane)
                v_dangerous_b_hook, vd_hperson_idx, dangerous_b_hook, d_hperson_idx = self.loc_console_with_arm_per(idx_co, cons_bb, hook_bbox, person_bbox)
                v_dangerous_b_hook = self.list_is_empty(v_dangerous_b_hook)
                dangerous_b_hook = self.list_is_empty(dangerous_b_hook)
                v_dangerous_b_excavator, vd_eperson_idx, dangerous_b_excavator, d_eperson_idx = self.loc_console_with_arm_per(idx_co, cons_bb, excavator_bbox, person_bbox)
                v_dangerous_b_excavator = self.list_is_empty(v_dangerous_b_excavator)
                dangerous_b_excavator = self.list_is_empty(dangerous_b_excavator)

                all_v_dangerous_b = self.conca(v_dangerous_b_crane, v_dangerous_b_hook, v_dangerous_b_excavator)
                all_dangerous_b = self.conca(dangerous_b_crane, dangerous_b_hook, dangerous_b_excavator)
                all_v_dangerous_b = self.list_is_empty(all_v_dangerous_b)
                all_dangerous_b = self.list_is_empty(all_dangerous_b)

                vd_cperson_idx = self.list_is_empty(vd_cperson_idx)
                vd_hperson_idx = self.list_is_empty(vd_hperson_idx)
                vd_eperson_idx = self.list_is_empty(vd_eperson_idx)

                d_cperson_idx = self.list_is_empty(d_cperson_idx)
                d_hperson_idx = self.list_is_empty(d_hperson_idx)
                d_eperson_idx = self.list_is_empty(d_eperson_idx)

                all_vd_idx = self.conca(vd_cperson_idx, vd_hperson_idx, vd_eperson_idx)
                all_d_idx = self.conca(d_cperson_idx, d_hperson_idx, d_eperson_idx)
                all_vd_idx = self.list_is_empty(all_vd_idx)
                all_d_idx = self.list_is_empty(all_d_idx)

                raw_v_dangerous_indx = self.add_result(all_vd_idx, raw_v_dangerous_indx)
                raw_dangerous_indx = self.add_result(all_d_idx, raw_dangerous_indx)
                raw_v_dangerous_b = self.add_result(all_v_dangerous_b, raw_v_dangerous_b)
                raw_dangerous_b = self.add_result(all_dangerous_b, raw_dangerous_b)
            self.add_p_person(p_bbox, raw_v_dangerous_b, raw_dangerous_b, raw_v_dangerous_indx, raw_dangerous_indx,
                         person_score, pr_result)
        return pr_result

    def add_print_result(self, box, score, re, de_type):
        """
        将某一个类别信息加入re中
        :param box:
        :param score:
        :param re:
        :param de_type:
        :return:
        """
        data = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'score': 0, 'type': 0}
        data["x1"] = float(box[1])
        data["y1"] = float(box[0])
        data["x2"] = float(box[3])
        data["y2"] = float(box[2])
        data["score"] = round(float(score), 4)
        data["type"] = de_type
        re.append(data)

    def scale_expand(self, coordinate, scale):
        """
        缩放坐标
        :param coordinate:
        :param scale:
        :return:
        """
        if coordinate == []:
            return None
        else:
            cy = (coordinate[0] + coordinate[2]) / 2
            cx = (coordinate[1] + coordinate[3]) / 2
            iw = abs((coordinate[0] - coordinate[2]) / 2)
            ih = abs((coordinate[1] - coordinate[3]) / 2)

            y_min = cy - scale * iw
            x_min = cy + scale * iw
            y_max = cx - scale * ih
            x_max = cx + scale * ih

            new_coor = [int(y_min), int(y_max), int(x_min), int(x_max)]
            return new_coor

    def loc_console_with_arm_per(self, con_idx, console_b, arm_bbox, per_box):
        con_xmin = console_b[1]
        con_xmax = console_b[3]
        con_xcenter = (console_b[1] + console_b[3]) / 2  # 控制台的x轴中心
        con_ymin = console_b[0]
        con_ymax = console_b[2]

        all_vd_idx_person = []
        all_d_idx_person = []
        v_dangerous_box = []
        dangerous_box = []

        if (not arm_bbox == []) and (not console_b == []):
            for idx_arm, coor_arm in enumerate(arm_bbox):
                ex_arm_ymin, ex_arm_xmin, ex_arm_ymax, ex_arm_xmax = self.scale_expand(coor_arm, 1.1)  # 将四个坐标放大1.1倍
                arm_xmax = coor_arm[3]
                arm_ymax = coor_arm[2]
                arm_xmin = coor_arm[1]
                arm_ymin = coor_arm[0]
                if con_xmin <= arm_xmax <= con_xmax and con_ymin <= arm_ymax <= con_ymax:  # arm in left/console in right

                    for idx_per, coor_per in enumerate(per_box):  # 获取person的box
                        per_xmax = coor_per[3]
                        per_ymax = coor_per[2]
                        per_xmin = coor_per[1]
                        per_ymin = coor_per[0]
                        per_xcenter = (per_xmax + per_xmin) / 2  # 获得person的中心
                        per_ycenter = (per_ymax + per_ymin) / 2

                        if arm_xmin <= per_xmax <= con_xmax and arm_ymin <= per_ymax <= con_ymax:  # this

                            all_vd_idx_person.append(idx_per)
                            v_dangerous_box.append(coor_per.tolist())

                        if arm_xmax <= per_xmin <= ex_arm_xmax and arm_ymin <= per_ymin <= ex_arm_ymax:
                            all_vd_idx_person.append(idx_per)
                            v_dangerous_box.append(coor_per.tolist())

                        condition1 = arm_xmax <= per_xcenter <= con_xmax and con_ymin <= per_ymax <= con_ymax
                        condition2 = arm_xmax <= per_xmin <= con_xmax and con_ymin <= per_ycenter <= con_ymax
                        condition3 = arm_xmax <= per_xcenter <= con_xmax and con_ymin <= per_ymin <= con_ymax

                        if condition1 or condition2 or condition3:
                            dangerous_box.append(coor_per.tolist())
                            all_d_idx_person.append(idx_per)

                if con_xcenter <= arm_xmin <= con_xmax and con_ymin <= arm_ymax <= con_ymax:  # arm in right/console in left
                    for idx_p, coor_p in enumerate(per_box):
                        per_xmax = coor_p[3]
                        per_ymax = coor_p[2]
                        per_xmin = coor_p[1]
                        per_ymin = coor_p[0]
                        per_xcent = (per_xmax + per_xmin) / 2
                        per_ycent = (per_ymax + per_ymin) / 2

                        if arm_xmin <= per_xmin <= arm_xmax and arm_ymin <= per_ymax <= con_ymax:
                            all_vd_idx_person.append(idx_p)
                            v_dangerous_box.append(coor_p.tolist())
                        if arm_xmax <= per_xmax <= ex_arm_xmax and arm_ymin <= per_ymin <= ex_arm_ymax:
                            all_vd_idx_person.append(idx_p)
                            v_dangerous_box.append(coor_p.tolist())

                        condition1 = con_xmin <= per_xcent <= arm_xmin and con_ymin <= per_ymax <= con_ymax
                        condition2 = con_xmin <= per_xmax <= arm_xmin and con_ymin <= per_ycent <= con_ymax
                        condition3 = con_xmin <= per_xcent <= arm_xmin and con_ymin <= per_ymin <= con_ymax

                        if condition1 or condition2 or condition3:
                            all_d_idx_person.append(idx_p)
                            dangerous_box.append(coor_p.tolist())

            return v_dangerous_box, all_vd_idx_person, dangerous_box, all_d_idx_person

        if (arm_bbox == []) and (not console_b == []):
            return None, None, None, None  # 表示没有操作臂

        if (not arm_bbox == []) and (console_b == []):
            for idx_arm, coor_arm in enumerate(arm_bbox):
                ex_arm_ymin, ex_arm_xmin, ex_arm_ymax, ex_arm_xmax = self.scale_expand_single(coor_arm, 1.5)
                for index_p, coordinate_p in enumerate(per_box):
                    per_xmax = coordinate_p[3]
                    per_ymax = coordinate_p[2]
                    per_xmin = coordinate_p[1]
                    per_ymin = coordinate_p[0]
                    p_x_cneter = (per_xmax + per_xmin) / 2
                    p_y_center = (per_ymin + per_ymax) / 2
                    if ex_arm_xmin <= p_x_cneter <= ex_arm_xmax and ex_arm_ymin <= p_y_center <= ex_arm_ymax:
                        all_vd_idx_person.append(index_p)
                        v_dangerous_box.append(coordinate_p.tolist())
            return v_dangerous_box, all_vd_idx_person, None, None
        else:
            return None, None, None, None

    def scale_expand_single(self, coordinate, scale):
        '''
        coordinate[0] ymin
        coordinate[1] xmin
        coordinate[2] ymax
        coordinate[3] xmax
        '''

        if coordinate == []:
            return None
        else:
            cy = (coordinate[0] + coordinate[2]) / 2
            iw = abs((coordinate[0] - coordinate[2]) / 2)
            x_min = cy + scale * iw
            new_c = [int(coordinate[0]), int(coordinate[1]), int(x_min), int(coordinate[3])]
            return new_c

    def list_is_empty(self, boundb):
        if boundb == []:
            return None
        elif boundb is None:
            return None
        else:
            return boundb

    def conca(self, v_danger_b_crane, v_danger_b_hook, v_danger_b_excavator):
        if v_danger_b_crane is not None and v_danger_b_hook is not None and v_danger_b_excavator is not None:
            re = np.vstack((v_danger_b_crane, v_danger_b_hook, v_danger_b_excavator))
            return re.tolist()
        elif v_danger_b_crane is None and v_danger_b_hook is not None and v_danger_b_excavator is not None:
            re = np.vstack((v_danger_b_hook, v_danger_b_excavator))
            return re.tolist()
        elif v_danger_b_crane is not None and v_danger_b_hook is None and v_danger_b_excavator is not None:
            re = np.vstack((v_danger_b_crane, v_danger_b_excavator))
            return re.tolist()
        elif v_danger_b_crane is not None and v_danger_b_hook is not None and v_danger_b_excavator is None:
            re = np.vstack((v_danger_b_crane, v_danger_b_hook))
            return re.tolist()
        elif v_danger_b_crane is not None and v_danger_b_hook is None and v_danger_b_excavator is None:
            re = v_danger_b_crane
            return re
        elif v_danger_b_crane is None and v_danger_b_hook is not None and v_danger_b_excavator is None:
            re = v_danger_b_hook
            return re
        elif v_danger_b_crane is None and v_danger_b_hook is None and v_danger_b_excavator is not None:
            re = v_danger_b_excavator
            return re
        else:
            return None

    def add_result(self, bb, ra):
        if bb is not None:
            for m in range(len(bb)):
                ra.append(bb[m])
            return ra
        else:
            return ra

    def add_p_person(self, personBox, vdBox, dBox, vdIdx, dIdx, person_score, pr):
        person_idx = list(range(0, len(personBox)))
        if vdIdx is not None:
            for j in range(len(vdIdx)):
                if vdIdx[j] in person_idx:
                    person_idx.remove(vdIdx[j])

        if dIdx is not None:
            for i in range(len(dIdx)):
                if dIdx[i] in person_idx:
                    person_idx.remove(dIdx[i])

        if vdIdx is not None:
            r = self.add_data(vdBox, vdIdx, person_score, pr, level=2)
        if dIdx is not None:
            r = self.add_data(dBox, dIdx, person_score, pr, level=1)
        if personBox is not None:
            r = self.add_data(personBox, person_idx, person_score, pr, level=0)
        return r

    def add_data(self, box, box_idx, s, pr, level):

        for i in range(len(box_idx)):
            data = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'score': 0, 'type': 0, 'risk_level': 0}
            a = box[i]
            b = box_idx[i]

            data['x1'] = float(a[1])
            data['y1'] = float(a[0])
            data['x2'] = float(a[3])
            data['y2'] = float(a[2])
            data['score'] = round(float(s[b]), 4)
            data['type'] = 'person'
            data['risk_level'] = level
            pr.append(data)

        return pr