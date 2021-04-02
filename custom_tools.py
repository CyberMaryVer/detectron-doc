import numpy as np
from PIL import ImageFont, ImageDraw, Image
import logging

import cv2
import csv
import os, time
import torch

from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from predictor import AsyncPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.file_io import PathManager
# from detectron2.utils.colormap import random_color

from geometry import draw_angles
from geometry import define_symmetry
from geometry import get_text_coords
from geometry import check_squats
from geometry import get_updated_keypoint_dict

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

_KEYPOINT_THRESHOLD = 0.05

INSTANCE_NUM = 0
KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

CONNECTION_RULES = [
    ('left_ear', 'left_eye', (255, 255, 255)),
    ('right_ear', 'right_eye', (255, 255, 255)),
    ('left_eye', 'nose', (255, 255, 255)),
    ('nose', 'right_eye', (255, 255, 255)),
    ('left_shoulder', 'right_shoulder', (0, 255, 55)),
    ('left_shoulder', 'left_elbow', (0, 255, 0)),
    ('right_shoulder', 'right_elbow', (0, 255, 0)),
    ('left_elbow', 'left_wrist', (0, 255, 0)),
    ('right_elbow', 'right_wrist', (0, 255, 0)),
    ('left_hip', 'right_hip', (0, 255, 55)),
    ('left_hip', 'left_knee', (0, 255, 0)),
    ('right_hip', 'right_knee', (0, 255, 0)),
    ('left_knee', 'left_ankle', (0, 255, 0)),
    ('right_knee', 'right_ankle', (0, 255, 0))
]


def create_csv_file(filename: str = "results.csv"):
    """
    creates csv file with keypoints names as column names
    """
    keypoint_names = KEYPOINT_NAMES
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["FRAME#", *keypoint_names, ';'])


def save_results_to_csv(results: dict, new_file: bool = True, filename: str = None):
    """
    saves results in csv file
    """
    if filename is None:
        filename = "results.csv"

    if new_file:
        create_csv_file(filename)

    with open(filename, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            all_coords = []
            for keypoint, coords in value.items():
                all_coords.append([coord.flatten().tolist()[0] for coord in coords])
            writer.writerow([key, *all_coords, ';'])


def create_keypoints_dictionary(outputs, cfg, instance=INSTANCE_NUM):
    """
    creates dictionary {keypoint name: keypoint coords}
    """
    keypoint_dict = {}
    keypoint_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).keypoint_names
    if len(outputs["instances"].pred_keypoints) == 0:
        return None
    try:
        keypoint_coords = outputs["instances"].pred_keypoints[instance]
    except IndexError:
        keypoint_coords = outputs["instances"].pred_keypoints[0]
    except Exception as e:
        print(e)
        return None

    for i in zip(keypoint_names, keypoint_coords):
        keypoint_dict.update({i[0]: i[1][:3, ]})

    return keypoint_dict


def test_from_file(path="instances_predictions.pth"):
    pred = torch.load(path)["instances"].to("cpu").pred_keypoints[INSTANCE_NUM]
    for K in KEYPOINT_NAMES:
        idx = KEYPOINT_NAMES.index(K)
        print(K, pred[idx].flatten().tolist())


def extract_predictions(predictions):
    pred = predictions["instances"].to("cpu").pred_keypoints[INSTANCE_NUM]
    extracted = {}
    for K in KEYPOINT_NAMES:
        idx = KEYPOINT_NAMES.index(K)
        extracted.update({K: pred[idx].flatten().tolist()})
    return extracted


def draw_text(img, text, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, position=(10, 10), font_scale=0.6,
              font_thickness=1, text_color=(0, 0, 0), text_color_bg=(255, 255, 255), alignment="center"):
    """
    draws text in a coloured rectangle
    """

    x, y = [int(x) for x in position]

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    # space
    text_w, text_h = [int(x) for x in text_size]
    cv2.rectangle(img, position, (x + text_w, y + text_h + 5), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h), font, font_scale, text_color, font_thickness)

    return img


def draw_angle_in_circle(img: object, angle: int, xy_coordinates: tuple, scale=None, symmetry=True):
    scale = 1 if scale is None else scale
    thickness = 1 if scale is None else int(scale * 1)
    shift = 0 if scale is None else int((len(str(angle)) == 3) * scale * 2.4)
    radius = 12 if scale is None else int(scale * 11)
    font_scale = .3 if scale is None else .3 * scale
    circle_color = (255, 255, 255)
    text_color = (0, 0, 0)

    if symmetry:
        radius = 16 if scale is None else int(scale * 15)
        font_scale = .3 if scale is None else .3 * scale
        circle_color = (0, 0, 255)

    x, y = xy_coordinates
    x1, y1 = x - int(8 * scale) - shift, y - int(6 * scale)
    x2, y2 = x - int(8 * scale) - shift, y + int(4 * scale)

    img = cv2.circle(img, (x, y), radius=radius, color=circle_color, thickness=-1)
    img = cv2.putText(img, f"{angle}", (x2, y2), cv2.FONT_HERSHEY_COMPLEX,
                      fontScale=font_scale,
                      color=text_color,
                      thickness=thickness)
    if symmetry:
        img = cv2.putText(img, f"+/-", (x1, y1), cv2.FONT_HERSHEY_COMPLEX,
                          fontScale=font_scale * .7,
                          color=(25, 100, 25),
                          thickness=1)

    return img


def draw_box_with_text(img, box_coord, text=None, edge_color=(0, 255, 0), linewidth=2):
    """
    draws box around
    """
    x0, y0, x1, y1 = box_coord
    print(x0, x1, y0, y1)
    width = x1 - x0
    height = y1 - y0

    img = cv2.rectangle(img, (x0, y0), (x1 + width, y1 + height), edge_color, linewidth)
    if text is not None:
        img = draw_text(img, text, text_color_bg=edge_color, position=(x0, y0))

    return img


def visualize_keypoints(keypoint_dict, im_wk, skeleton=1, side=None, mode=None, scale=None, threshold=None,
                        color_mode=None):
    if keypoint_dict is None:
        return im_wk

    if color_mode is None:
        skeleton_color1 = skeleton_color2 = (0, 255, 0)
        point_color1 = point_color2 = (255, 255, 255)
    else:
        skeleton_color1 = (0, 255, 0)
        skeleton_color2 = (188, 188, 188)
        point_color1 = (0, 255, 0)
        point_color2 = (188, 188, 188)

    rules = CONNECTION_RULES
    if side not in ["L", "R", None]:
        print("Error: wrong side parameter")
        return 1

    threshold = .5 if threshold is None else threshold

    if scale is not None:
        thickness = int(scale * 2.4)
        point_radius = int(scale * 6.4)
        fontscale = .3 * scale
    else:
        thickness = 2
        point_radius = 6
        fontscale = .3

    if skeleton != 0:

        for rule in rules:

            # set visibility for each side
            if side == "R":
                if "right" in rule[0] and "right" in rule[1]:
                    draw_connection = True
                else:
                    draw_connection = False
            elif side == "L":
                if "left" in rule[0] and "left" in rule[1]:
                    draw_connection = True
                else:
                    draw_connection = False
            else:
                if skeleton == 2:
                    rules_for_head = ["eye", "ear", "nose"]
                    draw_connection = True

                    for r in rule[:-1]:

                        if not r == "nose":
                            r = r.split("_")[1]

                        if r in rules_for_head:
                            draw_connection = False
                else:
                    draw_connection = True

            if draw_connection:
                try:
                    p1, p2, color = rule
                    color = skeleton_color1
                    x1, y1, t1 = keypoint_dict[p1]
                    x2, y2, t2 = keypoint_dict[p2]
                    if t1 < threshold and t2 < threshold:
                        color = skeleton_color2
                    im_wk = cv2.line(im_wk, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
                except:
                    pass

        # draw central line
        points_to_use = [('left_hip', 'right_hip'), ('left_shoulder', 'right_shoulder')]
        points_to_connect = []  # should be ['hip_centre, 'shoulder_centre', 'nose']

        for p in points_to_use:
            try:
                x1, y1, _ = keypoint_dict[p[0]]
                x2, y2, _ = keypoint_dict[p[1]]
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)
                points_to_connect.append((x, y))
            except:
                pass

        x, y, _ = keypoint_dict['nose']
        points_to_connect.append((int(x), int(y)))
        colors = (skeleton_color1, point_color1)
        im_wk = cv2.line(im_wk, points_to_connect[0], points_to_connect[1], colors[0], thickness=thickness)
        if skeleton == 1:
            im_wk = cv2.line(im_wk, points_to_connect[1], points_to_connect[2], colors[1], thickness=thickness)
        elif skeleton == 2:
            xrs, yrs, _ = keypoint_dict["right_shoulder"]
            xls, yls, _ = keypoint_dict["left_shoulder"]
            x1, y1, _ = keypoint_dict["nose"]
            x2, y2 = points_to_connect[1]
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            im_wk = cv2.line(im_wk, (int(xrs), int(yrs)), (x, y), colors[1], thickness=thickness)
            im_wk = cv2.line(im_wk, (int(xls), int(yls)), (x, y), colors[1], thickness=thickness)

        if side is not None:
            if side == "L":
                hip, shoulder = "left_hip", "left_shoulder"
            elif side == "R":
                hip, shoulder = "right_hip", "right_shoulder"
            else:
                hip = shoulder = None
            x1, y1, _ = keypoint_dict[hip]
            p1 = (int(x1), int(y1))
            x2, y2, _ = keypoint_dict[shoulder]
            p2 = (int(x2), int(y2))
            im_wk = cv2.line(im_wk, points_to_connect[0], p1, colors[0], thickness=thickness)
            im_wk = cv2.line(im_wk, points_to_connect[1], p2, colors[0], thickness=thickness)

    for name, i in keypoint_dict.items():
        # set visibility for each side
        if side == "R":
            if "right" in name:
                draw_point = True
            else:
                draw_point = False
        elif side == "L":
            if "left" in name:
                draw_point = True
            else:
                draw_point = False
        else:
            draw_point = True

            if skeleton == 2:
                rules_for_head = ["eye", "ear", "nose"]

                if not name == "nose":
                    name = name.split("_")[1]

                if name in rules_for_head:
                    draw_point = False

        if draw_point:
            x, y = int(i[0]), int(i[1])
            t = i[2]
            color = point_color1
            if t < threshold:
                color = point_color2
            im_wk = cv2.circle(im_wk,
                               (x, y),
                               radius=point_radius,
                               color=color,
                               thickness=-1)

            if mode == "keypoints_names":
                name = name.replace("_", " ")
                im_wk = draw_text(im_wk, name, font=cv2.FONT_HERSHEY_SIMPLEX, position=(x, y), font_scale=fontscale)

            elif mode == "angles":
                angles_dict = draw_angles(keypoint_dict, side=side)
                # angle_point_radius = point_radius * 2
                for kp, kp_data in angles_dict.items():
                    x, y = int(kp_data[1][0]), int(kp_data[1][1])
                    angle_value = int(kp_data[0])
                    im_wk = draw_angle_in_circle(im_wk, angle_value, (x, y), scale=scale, symmetry=False)
                    # x_, y_ = get_text_coords((x, y), scale)
                    # im_wk = cv2.circle(im_wk,
                    #                    (x, y),
                    #                    radius=angle_point_radius,
                    #                    color=(255, 255, 255),
                    #                    thickness=-1)
                    # im_wk = cv2.putText(im_wk, f"{kp_data[0]:.0f}",
                    #                     (x_, y_), cv2.FONT_HERSHEY_COMPLEX,
                    #                     fontScale=fontscale, color=(0, 0, 0), thickness=1)

    if mode == "symmetry" or mode == "squats":
        new_dict = get_updated_keypoint_dict(keypoint_dict)

        if mode == "symmetry":
            angles_dict = draw_angles(keypoint_dict)
            sym = define_symmetry(angles_dict)

            for key in sym.keys():
                # print(key)
                if "hip_center" in key:
                    key1 = key2 = "hip_center"
                elif "neck_center" in key:
                    key1 = key2 = "neck_center"
                else:
                    key1 = "right" + key
                    key2 = "left" + key
                # ±, º
                x1, y1, _ = new_dict[key1]
                x2, y2, _ = new_dict[key2]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                deviation = int(abs(sym[key]))
                im_wk = draw_angle_in_circle(im_wk, deviation, (x1, y1), scale=scale)
                im_wk = draw_angle_in_circle(im_wk, deviation, (x2, y2), scale=scale)

            if len(sym) > 0:
                print(sym)

        elif mode == "squats":
            check, coords = check_squats(new_dict)

            if check:
                color1 = (0, 0, 255)
                color2 = (0, 0, 255)
                thickness = thickness
            else:
                color1 = (0, 255, 0)
                color2 = (255, 255, 255)
                thickness = 1

            x1, y1, _ = [int(x) for x in coords[0]]
            x2, y2, _ = [int(x) for x in coords[1]]
            im_wk = cv2.circle(im_wk, (x1, y1), radius=point_radius * 2, color=color2, thickness=-1)
            im_wk = cv2.circle(im_wk, (x2, y2), radius=point_radius * 2, color=color2, thickness=-1)
            im_wk = cv2.line(im_wk, (x1 - 100, y1), (x1 + 100, y1), color=color1, thickness=thickness)
            im_wk = cv2.line(im_wk, (x2 - 100, y2), (x2 + 100, y2), color=color1, thickness=thickness)

    return im_wk


class DetectronVideo:
    def __init__(self, path_to_video, cfg, parallel=False, instance=0):
        self.instance = instance

        # general parameters
        self.cfg = cfg
        self.video = cv2.VideoCapture(path_to_video)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale = max(self.height, self.width) / 600
        self.frames_per_second = self.video.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.basename = os.path.basename(path_to_video)
        self.tracking = {}
        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

        # debug parameters
        self.drop_frame_interval = 1
        self.num_frames_for_debug = self.num_frames

    def _frame_from_video(self, video):
        # while video.isOpened():
        f = 0
        while f < self.num_frames_for_debug:
            success, frame = video.read()
            if success:
                yield frame
                f += 1
            else:
                break

    def run_on_video(self, debug_params=None, side=None, skeleton=True, mode=None,
                     save_output=False, save_path=None):
        output_file = None
        if save_output:
            if save_path is None:
                output_filename = os.path.join(self.basename.split('.')[0] + "_out.mp4")
            else:
                output_filename = save_path

            output_file = cv2.VideoWriter(
                filename=output_filename,
                # some installation of opencv may not support mp4v (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=float(self.frames_per_second),
                frameSize=(self.width, self.height),
                isColor=True,
            )

        if debug_params is not None:
            print(f"::Debug mode::\ncalculation step = {debug_params[0]}"
                  f"\nnumber of frames = {debug_params[1]}")
            self.drop_frame_interval = debug_params[0]
            self.num_frames_for_debug = debug_params[1]

        frame_gen = self._frame_from_video(self.video)
        current_frame = 0

        read_one_frame = True
        while read_one_frame:
            _, initial_frame = self.video.read()
            read_one_frame = False

        keypoint_dict = None
        # pred = self.predictor(initial_frame)
        # keypoint_dict = create_keypoints_dictionary(pred, self.cfg, self.instance)

        for frame in frame_gen:

            if current_frame % self.drop_frame_interval == 0:
                # start = time.time()
                pred = self.predictor(frame)
                # with PathManager.open("instances_predictions.pth", "ab") as f:
                #     torch.save(pred, f)
                keypoint_dict = create_keypoints_dictionary(pred, self.cfg, self.instance)
                # print(f"\ntime (pred): {time.time() - start}")
                self.tracking.update({current_frame: keypoint_dict})

            frame = visualize_keypoints(keypoint_dict, frame, skeleton=skeleton, side=side, mode=mode,
                                        scale=self.scale)

            if save_output:
                output_file.write(frame)

            yield (frame, keypoint_dict)

            current_frame += 1

        self.video.release()
        if save_output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # test_from_file()
    im = cv2.imread("tests/test.jpg")
    im = draw_box_with_text(im, (1, 1, 100, 100), "Oh my god! It is almost 6:00 a.m.")
    cv2.imshow("", im)
    cv2.waitKey(0)
