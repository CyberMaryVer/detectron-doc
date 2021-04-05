import mediapipe as mp
import cv2.cv2 as cv2
import numpy as np
import os

from geometry import get_central_points, define_symmetry, center_of_gravity, get_angle_dict
from custom_tools import CONNECTION_RULES, CONNECTION_RULES_ADDITION1, \
    CONNECTION_RULES_ADDITION2, CONNECTION_RULES_ADDITION3
from custom_tools import draw_text, draw_angle_in_circle

_KEYPOINT_THRESHOLD = .5


def get_updated_keypoint_dict(keypoint_dict):
    points_to_connect = get_central_points(keypoint_dict)
    keypoint_dict.update(points_to_connect)
    return keypoint_dict


def draw_skeleton(img, keypoints, side=None, threshold=0., thickness=1,
                  headless=False, draw_invisible=False, color_invisible=(188, 188, 188)):
    """
    This function draws connections on the image and returns image
    """
    rules = CONNECTION_RULES
    additional_rules = CONNECTION_RULES_ADDITION1
    if not headless:
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
        rules.append(additional_rules[0])
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


def draw_joints(img, keypoints, threshold=0., side=None, headless=False, color=(255, 255, 255),
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


def visualize_keypoints_mp(keypoint_dict, im_wk, skeleton=1, side=None, mode=None, scale=None, threshold=None,
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
                angles_dict = get_angle_dict(keypoint_dict, side=side, dict_is_updated=True)

                for kp, kp_data in angles_dict.items():
                    x, y = int(kp_data[1][0]), int(kp_data[1][1])
                    angle_value = kp_data[0]
                    im_wk = draw_angle_in_circle(im_wk, angle_value, (x, y), scale=scale, symmetry=False)

            elif mode == "symmetry" and visibility:
                # """displays symmetry breaks"""
                angles_dict = get_angle_dict(keypoint_dict, dict_is_updated=True)
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
                line_length = max(im_wk.shape) // 3

                # draw lines
                im_wk = cv2.line(im_wk, (x1_, y1_), (x2_, y2_), color=line_color, thickness=1, lineType=cv2.LINE_AA)
                im_wk = cv2.line(im_wk, (x2 - line_length, y1), (x1 + line_length, y1), color=line_color, thickness=1,
                                 lineType=cv2.LINE_AA)
                # draw points
                im_wk = cv2.circle(im_wk, (x1, y1), radius=point_radius * 2, color=point_color, thickness=-1,
                                   lineType=cv2.LINE_AA)
                im_wk = cv2.circle(im_wk, (x2, y2), radius=point_radius, color=(255, 255, 255), thickness=-1,
                                   lineType=cv2.LINE_AA)

    return im_wk


# def mediapipe_predictor(img, draw=False, threshold=.8):
#     mp_pose = mp.solutions.pose
#
#     # Prepare DrawingSpec for drawing the face landmarks later.
#     mp_drawing = mp.solutions.drawing_utils
#     drawing_spec1 = mp_drawing.DrawingSpec(thickness=-1, circle_radius=4, color=(255, 255, 255))  # for keypoints
#     drawing_spec2 = mp_drawing.DrawingSpec(thickness=1, circle_radius=4, color=(0, 255, 0))  # for lines
#
#     with mp_pose.Pose(static_image_mode=True, min_detection_confidence=threshold) as pose:
#         # Convert the BGR image to RGB and process it with MediaPipe Pose.
#         results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         if results.pose_landmarks is None:
#             return
#         image_hight, image_width, _ = img.shape
#
#     if draw:
#         # Draw pose landmarks.
#         print(f'Pose landmarks:')
#         annotated_image = img.copy()
#         mp_drawing.draw_landmarks(
#             image=annotated_image,
#             landmark_list=results.pose_landmarks,
#             connections=mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=drawing_spec1,
#             connection_drawing_spec=drawing_spec2)
#         cv2.imshow("", annotated_image)
#         cv2.waitKey(0)
#
#     try:
#         NOSE = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
#         LEFT_KNEE = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
#         RIGHT_KNEE = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
#         LEFT_HIP = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
#         RIGHT_HIP = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
#         LEFT_ELBOW = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
#         RIGHT_ELBOW = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
#         LEFT_WRIST = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
#         RIGHT_WRIST = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
#         LEFT_SHOULDER = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#         RIGHT_SHOULDER = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#         LEFT_EYE = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
#         RIGHT_EYE = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
#         LEFT_EAR = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
#         RIGHT_EAR = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
#         LEFT_ANKLE = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
#         RIGHT_ANKLE = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
#     except Exception as e:
#         print(e)
#         return
#
#     COCO_KEYPOINT_NAMES = {'nose': NOSE,
#                            'left_eye': LEFT_EYE,
#                            'right_eye': RIGHT_EYE,
#                            'left_ear': LEFT_EAR,
#                            'right_ear': RIGHT_EAR,
#                            'left_shoulder': LEFT_SHOULDER,
#                            'right_shoulder': RIGHT_SHOULDER,
#                            'left_elbow': LEFT_ELBOW,
#                            'right_elbow': RIGHT_ELBOW,
#                            'left_wrist': LEFT_WRIST,
#                            'right_wrist': RIGHT_WRIST,
#                            'left_hip': LEFT_HIP,
#                            'right_hip': RIGHT_HIP,
#                            'left_knee': LEFT_KNEE,
#                            'right_knee': RIGHT_KNEE,
#                            'left_ankle': LEFT_ANKLE,
#                            'right_ankle': RIGHT_ANKLE}
#
#     def get_coords(keypoint, img_shape=None, with_z=True):
#         """get coords for keypoint by name"""
#         x, y, z, vis = keypoint.x, keypoint.y, keypoint.z, keypoint.visibility
#
#         if img_shape is not None:
#             image_hight, image_width, _ = img_shape
#             image_depth = (image_hight + image_width) / 2
#             x, y, z = int(x * image_width), int(y * image_hight), int(z * image_depth)
#
#         if not with_z:
#             return x, y, vis
#
#         return x, y, z, vis
#
#     def get_keypoints_dict(img_shape):
#         keypoints_dict = {}
#         for name, obj in COCO_KEYPOINT_NAMES.items():
#             keypoints_dict.update({name: get_coords(obj, img_shape, False)})
#         return keypoints_dict
#
#     shape = np.array(img).shape
#     keypoints = get_keypoints_dict(shape)
#     return keypoints


class MpipePredictor(mp.solutions.pose.Pose):
    def __init__(self, detection_thr, tracking_thr=.99, path_to_video=None, static=False, instance=0):
        super().__init__(static_image_mode=static, min_detection_confidence=detection_thr,
                         min_tracking_confidence=tracking_thr)
        self.instance = instance
        self.path_to_video = path_to_video
        if self.path_to_video is not None:
            self.video = cv2.VideoCapture(self.path_to_video)
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.scale = max(self.height, self.width) / 850
            self.frames_per_second = self.video.get(cv2.CAP_PROP_FPS)
            self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.basename = os.path.basename(self.path_to_video)
        self.keypoints = {}
        self.tracking = {}

    def run_on_video(self, side=None, skeleton=1, mode=None, joints=True, threshold=.5, color_mode=None,
                     draw_invisible=False, save_output=False, save_path=None, debug_params=None):
        if self.path_to_video is None:
            return

        output_file = None

        with self:
            if save_output:
                if save_path is None:
                    output_filename = os.path.join(self.basename.split('.')[0] + "_out.mp4")
                else:
                    output_filename = save_path

                output_file = cv2.VideoWriter(
                    filename=output_filename,
                    fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                    fps=float(self.frames_per_second),
                    frameSize=(self.width, self.height),
                    isColor=True,)

            frame_gen = self._frame_from_video(self.video)
            for frame in frame_gen:
                keypoints = self.get_keypoints(frame)
                frame = visualize_keypoints_mp(keypoints, frame, skeleton=skeleton, side=side, mode=mode,
                                               scale=self.scale, threshold=threshold, color_mode=color_mode,
                                               draw_invisible=draw_invisible, joints=joints, dict_is_updated=False)
                if save_output:
                    output_file.write(frame)

                yield frame, keypoints

            self.video.release()

            if save_output:
                output_file.release()
            else:
                cv2.destroyAllWindows()

    def get_keypoints(self, img, get3d=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        results = self.process(img)
        if results is None:
            return
        idxs = mp.solutions.pose.PoseLandmark
        self.keypoints = {'nose': results.pose_landmarks.landmark[idxs.NOSE],
                          'left_eye': results.pose_landmarks.landmark[idxs.LEFT_EYE],
                          'right_eye': results.pose_landmarks.landmark[idxs.RIGHT_EYE],
                          'left_ear': results.pose_landmarks.landmark[idxs.LEFT_EAR],
                          'right_ear': results.pose_landmarks.landmark[idxs.RIGHT_EAR],
                          'left_shoulder': results.pose_landmarks.landmark[idxs.LEFT_SHOULDER],
                          'right_shoulder': results.pose_landmarks.landmark[idxs.RIGHT_SHOULDER],
                          'left_elbow': results.pose_landmarks.landmark[idxs.LEFT_ELBOW],
                          'right_elbow': results.pose_landmarks.landmark[idxs.RIGHT_ELBOW],
                          'left_wrist': results.pose_landmarks.landmark[idxs.LEFT_WRIST],
                          'right_wrist': results.pose_landmarks.landmark[idxs.RIGHT_WRIST],
                          'left_hip': results.pose_landmarks.landmark[idxs.LEFT_HIP],
                          'right_hip': results.pose_landmarks.landmark[idxs.RIGHT_HIP],
                          'left_knee': results.pose_landmarks.landmark[idxs.LEFT_KNEE],
                          'right_knee': results.pose_landmarks.landmark[idxs.RIGHT_KNEE],
                          'left_ankle': results.pose_landmarks.landmark[idxs.LEFT_ANKLE],
                          'right_ankle': results.pose_landmarks.landmark[idxs.RIGHT_ANKLE]}
        keypoints_dict = {}
        for name, obj in self.keypoints.items():
            keypoints_dict.update({name: self._get_coords(obj, img.shape, get3d)})
        return keypoints_dict

    def _get_coords(self, keypoint, img_shape=None, with_z=True):
        """get coords for keypoint"""
        x, y, z, vis = keypoint.x, keypoint.y, keypoint.z, keypoint.visibility

        if img_shape is not None:
            image_height, image_width, _ = img_shape
            image_depth = (image_height + image_width) / 2
            x, y, z = int(x * image_width), int(y * image_height), int(z * image_depth)

        if not with_z:
            return x, y, vis

        return x, y, z, vis

    def _frame_from_video(self, video):
        # while video.isOpened():
        f = 0
        while f < self.num_frames:
            success, frame = video.read()
            if success:
                yield frame
                f += 1
            else:
                break


if __name__ == "__main__":

    _SCALE = .8
    video = MpipePredictor(path_to_video="tests/s.mp4", detection_thr=.7, tracking_thr=.7)

    for vis_frame, _ in video.run_on_video(side="L", skeleton=2, mode="gravity_center", save_output=True,
                                           save_path="asdf.mp4"):

        cv2.namedWindow(video.basename, cv2.WINDOW_AUTOSIZE)
        video_width, video_height = int(video.width * _SCALE), int(video.height * _SCALE)
        generated_frame = cv2.resize(vis_frame, (video_width, video_height))
        cv2.imshow(video.basename, generated_frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    predictor = MpipePredictor(detection_thr=.8, tracking_thr=.9)
    im = cv2.imread("tests/test.jpg")
    kps = predictor.get_keypoints(im)
    print(kps)
