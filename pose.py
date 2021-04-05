from inference_config import *
from custom_tools import draw_text
import multiprocessing as mp
import cv2, os, tqdm
from time import time
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from custom_tools import create_keypoints_dictionary
from custom_tools import visualize_keypoints
from custom_tools import save_results_to_csv
from custom_tools import DetectronVideo
from geometry import convert_dict

mp.set_start_method("spawn", force=True)
args = get_parser().parse_args()
print(args)

# choose inference parameters
MODEL = PATHS[args.config_file]
DEVICE = args.device
THRESHOLD = args.threshold

# set paths to video and photo
_IMAGE = args.image
_VIDEO = args.video
_WEBCAM = args.webcam
_TEST = args.test

# settings for the test script
_SAVE_OUTPUT = args.output
_SAVE_FILE_NAME = args.output_name
_DEBUG_VIDEO = args.debug_video
_DEBUG_PATH_TO_VIDEO = "tests/s.mp4"

_TRACKING_DEBUG = True

# visualisation parameters
_INSTANCE_NUM = args.instance
_CONNECTIONS = args.skeleton
_SIDE = args.side
_MODE = args.mode  # ["keypoints_names", "angles", "symmetry"]
_CSV = args.csv

# real time correction
_EXERCISE = args.exercise # ["squats", "plank", "weight_lifting"]
_SCALE = args.scale

if __name__ == "__main__":
    cfg = build_cfg(MODEL, DEVICE, THRESHOLD)
    predictor = DefaultPredictor(cfg)

    if args.debug_video is not None:
        debug_params = [int(x) for x in args.debug_video]
    else:
        debug_params = None

    if _EXERCISE is not None and _VIDEO is not None:
        _DEBUG_PATH_TO_VIDEO = _VIDEO
        _VIDEO = None

    if _IMAGE is not None and _VIDEO is not None:
        print("You indicated both --image and --video parameter. Inference will be done only for image. "
              "Please run inference on video separately")

    # test outputs and draw custom prediction
    if _IMAGE is not None:

        if _TEST:
            im = cv2.imread(_IMAGE)
            im_wk = np.asarray(im.copy())
            start_time = time()
            outputs = predictor(im)
            output_name = _IMAGE[:-4] + '_inference_out.jpg'

            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            end_time = time()
            time_txt = f'Time of inference on {DEVICE} = {(end_time - start_time):.2f} sec'
            print(time_txt)
            output_image = out.get_image()[:, :, ::-1]
            output_image = np.asarray(output_image.copy())
            output_image = draw_text(output_image, time_txt, font_scale=.8, text_color=(0, 0, 0),
                                     text_color_bg=(0, 255, 0))

        else:
            im = cv2.imread(_IMAGE)
            im_wk = np.asarray(im.copy())
            scale = max(im_wk.shape) / 600
            outputs = predictor(im)
            output_name = _IMAGE[:-4] + '_visualisation_out.jpg'

            keypoint_dict = create_keypoints_dictionary(outputs, cfg, _INSTANCE_NUM)
            output_image = visualize_keypoints(keypoint_dict, im_wk, _CONNECTIONS, _SIDE, _MODE, scale)

        cv2.imshow(output_name, output_image)
        cv2.waitKey(0)
        cv2.imwrite(output_name, output_image)

    elif _VIDEO is not None:
        save_path = _SAVE_FILE_NAME if _SAVE_FILE_NAME is not None else None

        video = DetectronVideo(_VIDEO, cfg)
        total = video.num_frames if debug_params is None else debug_params[1]
        for vis_frame, _ in tqdm.tqdm(video.run_on_video(debug_params=debug_params,
                                                         side=_SIDE,
                                                         skeleton=_CONNECTIONS,
                                                         mode=_MODE,
                                                         save_output=_SAVE_OUTPUT,
                                                         save_path=save_path),
                                      total=total):

            cv2.namedWindow(video.basename, cv2.WINDOW_AUTOSIZE)
            video_width, video_height = int(video.width * _SCALE), int(video.height * _SCALE)
            generated_frame = cv2.resize(vis_frame, (video_width, video_height))
            cv2.imshow(video.basename, generated_frame)
            if cv2.waitKey(1) == 27:
                break  # esc to quit

        if _CSV:
            results = video.tracking
            save_results_to_csv(results)

        # if _TRACKING_DEBUG:
        #     import pickle as pk
        #     with open("tracking.pickle", "wb") as f:
        #         pk.dump(video.tracking, f)
        #     print(len(video.tracking), video.tracking[0])

    elif _EXERCISE is not None:

        task_finished = False
        output_video = None
        keypoint_sets = []
        frame = 0

        if _WEBCAM:
            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            while cv2.waitKey(1) != 27:
                ret, frame = cam.read()

                outputs = predictor(frame)

                keypoint_dict = create_keypoints_dictionary(outputs, cfg, _INSTANCE_NUM)
                visualize_keypoints(keypoint_dict, frame, _CONNECTIONS, _SIDE, _MODE)

                cv2.imshow('frame', frame)

            cam.release()
            cv2.destroyAllWindows()

        else:
            save_path = _SAVE_FILE_NAME if _SAVE_FILE_NAME is not None else None

            video = DetectronVideo(_DEBUG_PATH_TO_VIDEO, cfg)
            total = video.num_frames if debug_params is None else debug_params[1]
            for vis_frame, keypoint_dict in tqdm.tqdm(video.run_on_video(debug_params=debug_params,
                                                                         side=_SIDE,
                                                                         skeleton=_CONNECTIONS,
                                                                         mode=_MODE,
                                                                         save_output=_SAVE_OUTPUT,
                                                                         save_path=save_path),
                                                      total=total):

                cv2.namedWindow(video.basename, cv2.WINDOW_AUTOSIZE)
                video_width, video_height = int(video.width * _SCALE), int(video.height * _SCALE)
                generated_frame = cv2.resize(vis_frame, (video_width, video_height))
                cv2.imshow(video.basename, generated_frame)

                if cv2.waitKey(1) == 27:
                    task_finished = True
                    break  # esc to quit

            cv2.destroyAllWindows()

            if _CSV:
                results = video.tracking
                save_results_to_csv(results)

    elif _TEST:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # cv2.OPENCV_VIDEOIO_DEBUG = 0

        while cv2.waitKey(1) != 27:
            ret, frame = cam.read()
            print(ret)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', gray)

        cam.release()
        cv2.destroyAllWindows()

    else:
        print("Run module with --help parameter to learn about the usage")
