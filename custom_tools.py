import numpy as np
# from PIL import ImageFont, ImageDraw, Image
# import logging

import cv2
import csv
import os, time

# import torch #########################
# from detectron2.data import MetadataCatalog #########################
# from detectron2.engine import DefaultPredictor #########################
# from predictor import AsyncPredictor #########################

from geometry import get_angle_dict
from geometry import define_symmetry
from geometry import center_of_gravity
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
    ('left_ear', 'left_eye', (255, 255, 255)),  # 0
    ('right_ear', 'right_eye', (255, 255, 255)),  # 1
    ('left_eye', 'nose', (255, 255, 255)),  # 2
    ('nose', 'right_eye', (255, 255, 255)),  # 3
    ('left_shoulder', 'right_shoulder', (0, 255, 55)),  # 4
    ('left_shoulder', 'left_elbow', (0, 255, 0)),  # 5
    ('right_shoulder', 'right_elbow', (0, 255, 0)),  # 6
    ('left_elbow', 'left_wrist', (0, 255, 0)),  # 7
    ('right_elbow', 'right_wrist', (0, 255, 0)),  # 8
    ('left_hip', 'right_hip', (0, 255, 55)),  # 9
    ('left_hip', 'left_knee', (0, 255, 0)),  # 10
    ('right_hip', 'right_knee', (0, 255, 0)),  # 11
    ('left_knee', 'left_ankle', (0, 255, 0)),  # 12
    ('right_knee', 'right_ankle', (0, 255, 0))  # 13
]
CONNECTION_RULES_ADDITION1 = [
    ('neck_center', 'hip_center', (0, 255, 0)),
    ('neck_center', 'nose', (0, 255, 0)),
]

CONNECTION_RULES_ADDITION2 = [
    ('head_center', 'left_shoulder', (255, 255, 255)),
    ('head_center', 'right_shoulder', (255, 255, 255)),
    ('head_center', 'neck_center', (255, 255, 255))
]

CONNECTION_RULES_ADDITION3 = [
    ('hip_center', 'left_hip', (0, 255, 0)),
    ('hip_center', 'right_hip', (0, 255, 0)),
    ('neck_center', 'left_shoulder', (0, 255, 0)),
    ('neck_center', 'right_shoulder', (0, 255, 0)),
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


def draw_skeleton(img, keypoints, side=None, threshold=0, thickness=1,
                  headless=False, draw_invisible=False, color_invisible=(188, 188, 188)):
    """
    This function draws connections on the image and returns image
    """
    rules = CONNECTION_RULES
    additional_rules = CONNECTION_RULES_ADDITION1
    for addition in additional_rules:
        rules.append(addition)

    if len(keypoints) == 17:
        keypoints = get_updated_keypoint_dict(keypoints)

    if headless:
        xrs, yrs, _ = keypoints["right_shoulder"]
        xls, yls, _ = keypoints["left_shoulder"]
        x1, y1, t1 = keypoints["nose"]
        x2, y2, t2 = keypoints["neck_center"]
        # calculate point to connect the lines
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        t = (t1 + t2) / 2
        keypoints.update({"head_center": (x, y, t)})
        additional_rules = CONNECTION_RULES_ADDITION2
        rules = rules[4:-1]
        for addition in additional_rules:
            rules.append(addition)

    if side is not None:
        additional_rules = CONNECTION_RULES_ADDITION3
        for addition in additional_rules:
            rules.append(addition)

        if side == "L":
            side, other = "left", "right"
        else:
            side, other = "right", "left"

    for rule in rules:
        # set visibility
        draw_connection = True

        if side is not None:
            # print(side, rule[0], rule[1], side in rule[0] and side in rule[1])
            if side in rule[0] and side in rule[1]:
                draw_connection = True
            elif "center" in rule[0]:
                draw_connection = True
                if other in rule[1]:
                    draw_connection = False
            else:
                draw_connection = False

        if draw_connection:
            try:
                p1, p2, color = rule
                x1, y1, t1 = keypoints[p1]
                x2, y2, t2 = keypoints[p2]

                if t1 > threshold and t2 > threshold:
                    img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color,
                                   thickness=thickness, lineType=cv2.LINE_AA)
                elif draw_invisible:
                    color = color_invisible
                    img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color,
                                   thickness=thickness, lineType=cv2.LINE_AA)
                else:
                    pass
            except Exception as e:
                print("Draw connection error:", e)
                pass

    return img


def draw_joints(img, keypoints, threshold=0, side=None, headless=False, color=(255, 255, 255),
                draw_invisible=False, color_invisible=(188, 188, 188), point_radius=6):
    """
    This function draws keypoints on the image and returns image
    """
    if len(keypoints) == 17:
        keypoints = get_updated_keypoint_dict(keypoints)

    if side is not None:
        side = "right" if side == "R" else "left"

    keypoints_visibility = {x: True for x in keypoints.keys()}

    for name, i in keypoints.items():
        # set visibility for each side
        draw_point = True

        if side is not None and side not in name:
            draw_point = False
            keypoints_visibility.update({name: draw_point})

        if headless:
            rules_for_head = ["eye", "ear", "nose"]

            if not name == "nose":
                name = name.split("_")[1]

            if name in rules_for_head:
                draw_point = False
                keypoints_visibility.update({name: draw_point})

        if draw_point:
            x, y, t = int(i[0]), int(i[1]), i[2]
            if t > threshold:
                img = cv2.circle(img, (x, y), radius=point_radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
            elif t < threshold and draw_invisible:
                color = color_invisible
                img = cv2.circle(img, (x, y), radius=point_radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
            else:
                pass

    return img, keypoints_visibility


def draw_text(img, text, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, position=(10, 10), font_scale=0.6,
              font_thickness=1, text_color=(0, 0, 0), text_color_bg=(255, 255, 255), alignment="center"):
    """
    draws text in a coloured rectangle
    """

    x, y = [int(x) for x in position]

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    # space
    text_w, text_h = [int(x) for x in text_size]
    cv2.rectangle(img, position, (x + text_w, y + text_h + 5), text_color_bg, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, text, (x, y + text_h), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

    return img


def draw_angle_in_circle(img: object, angle: int, xy_coordinates: tuple, scale=None, symmetry=True):
    """
    draws angle value in a circle
    """
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

    img = cv2.circle(img, (x, y), radius=radius, color=circle_color, thickness=-1, lineType=cv2.LINE_AA)
    img = cv2.putText(img, f"{angle}", (x2, y2), cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale,
                      color=text_color, thickness=thickness, lineType=cv2.LINE_AA)
    if symmetry:
        img = cv2.putText(img, f"+/-", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale * .7,
                          color=(25, 100, 25), thickness=1, lineType=cv2.LINE_AA)

    return img


def draw_box_with_text(img, text=None, edge_color=(0, 255, 0), border=2, mode=0):
    """
    draws box around
    """
    width, height = img.shape[1::-1]
    scale = max(width, height) / 400
    font_scale, font_thickness = .4 * scale, int(scale)

    if mode == 0:  # standard mode
        img = cv2.copyMakeBorder(img, border + 18, border, border, border, cv2.BORDER_CONSTANT, value=edge_color)

    elif mode == 1:  # low vision
        img = cv2.copyMakeBorder(img, height // 3, border, border, border, cv2.BORDER_CONSTANT, value=edge_color)
        font_scale, font_thickness = 1.4 * scale, int(2 * scale)

    if text is not None:
        x = y = border
        img = draw_text(img, text, text_color_bg=edge_color, position=(x + 2, y + 2), font_scale=font_scale,
                        font_thickness=font_thickness)

    return img


def visualize_keypoints(keypoint_dict, im_wk, skeleton=1, side=None, mode=None, scale=None, threshold=None,
                        color_mode=None, draw_invisible=False, joints=True, dict_is_updated=False):
    if keypoint_dict is None:
        return im_wk
    if side not in ["L", "R", None]:
        print("Error: wrong side parameter")
        return 1

    if not dict_is_updated:  # for compability with mediapipe predictor
        all_keypoints = get_updated_keypoint_dict(keypoint_dict)
    else:
        all_keypoints = keypoint_dict

    threshold = _KEYPOINT_THRESHOLD if threshold is None else threshold

    if color_mode is None:
        point_color1 = point_color2 = (255, 255, 255)
    else:
        point_color1 = (0, 255, 0)

    if scale is not None:
        thickness = int(scale * 2.4)
        point_radius = int(scale * 6.4)
        font_scale = .3 * scale
    else:
        thickness = 2
        point_radius = 6
        font_scale = .3

    if skeleton != 0:
        headless = True if skeleton == 2 else False
        im_wk = draw_skeleton(im_wk, all_keypoints, side=side, threshold=threshold, thickness=thickness,
                              headless=headless, draw_invisible=draw_invisible)

    if joints:
        headless = True if skeleton == 2 else False
        im_wk, vis = draw_joints(im_wk, all_keypoints, threshold=threshold, side=side, headless=headless,
                                 color=point_color1, draw_invisible=draw_invisible, point_radius=point_radius)

        for name, visibility in vis.items():

            if mode == "keypoints_names" and visibility:
                # """displays keypoint names"""
                x, y, _ = all_keypoints[name]
                name = name.replace("_", " ")
                print(x, y)
                im_wk = draw_text(im_wk, name, font=cv2.FONT_HERSHEY_SIMPLEX, position=(int(x), int(y)),
                                  font_scale=font_scale)

            elif mode == "angles" and visibility:
                # """displays angles values"""
                angles_dict = get_angle_dict(keypoint_dict, side=side)

                for kp, kp_data in angles_dict.items():
                    x, y = int(kp_data[1][0]), int(kp_data[1][1])
                    angle_value = kp_data[0]
                    im_wk = draw_angle_in_circle(im_wk, angle_value, (x, y), scale=scale, symmetry=False)

            elif mode == "symmetry" and visibility:
                # """displays symmetry breaks"""
                angles_dict = get_angle_dict(keypoint_dict)
                sym = define_symmetry(angles_dict)

                for key in sym.keys():
                    if "hip_center" in key:
                        key1 = key2 = "hip_center"
                    elif "neck_center" in key:
                        key1 = key2 = "neck_center"
                    else:
                        key1 = "right" + key
                        key2 = "left" + key
                    # ±, º
                    x1, y1, _ = all_keypoints[key1]
                    x2, y2, _ = all_keypoints[key2]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    deviation = int(abs(sym[key]))
                    im_wk = draw_angle_in_circle(im_wk, deviation, (x1, y1), scale=scale)
                    im_wk = draw_angle_in_circle(im_wk, deviation, (x2, y2), scale=scale)

                if len(sym) > 0:
                    print(sym)

            elif mode == "gravity_center":
                # """displays center of gravity"""
                xline, yline = center_of_gravity(all_keypoints)
                x1, y1 = [int(p) for p in xline]
                x2, y2 = [int(p) for p in yline]
                div = 1.0 * (x2 - x1) if x2 != x1 else .00001
                a = (1.0 * (y2 - y1)) / div
                b = -a * x1 + y1
                y1_, y2_ = 0, im_wk.shape[1]
                x1_ = int((y1_ - b) / a)
                x2_ = int((y2_ - b) / a)
                line_color = (255, 100, 100)
                point_color = (255, 100, 100)
                line_length = max(im_wk.shape)//3

                # draw lines
                im_wk = cv2.line(im_wk, (x1_, y1_), (x2_, y2_), color=line_color, thickness=1, lineType=cv2.LINE_AA)
                im_wk = cv2.line(im_wk, (x2 - line_length, y1), (x1 + line_length, y1), color=line_color, thickness=1,
                                 lineType=cv2.LINE_AA)
                # draw points
                im_wk = cv2.circle(im_wk, (x1, y1), radius=point_radius * 2, color=point_color, thickness=-1,
                                   lineType=cv2.LINE_AA)
                im_wk = cv2.circle(im_wk, (x2, y2), radius=point_radius, color=(255,255,255), thickness=-1,
                                   lineType=cv2.LINE_AA)

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

            yield frame, keypoint_dict

            current_frame += 1

        self.video.release()
        if save_output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # test_from_file()
    im_ = cv2.imread("tests/test.jpg")
    # im = draw_box_with_text(im, "Very good!", mode=0, edge_color=(255, 0, 0))
    from sandbox import ttt

    keyps = get_updated_keypoint_dict(ttt)

    im = draw_skeleton(im_, ttt, thickness=2, headless=True, threshold=0)
    im, _ = draw_joints(im, ttt, headless=True, threshold=0)
    cv2.imshow("", im)
    cv2.waitKey(0)
