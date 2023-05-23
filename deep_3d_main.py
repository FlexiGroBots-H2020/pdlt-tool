# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import shutil

import numpy as np
import os
import time
import cv2
import tqdm
from DPT.run_monodepth_single import DeepEstimator
from deep_3d_utils import detic2flat, xyxy2tlwh, update_tracks, detections_depth, correct_distance_calibration
import torch

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools_clip import generate_clip_detections as gdet
import clip

import sys
sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from Detic.third_party.CenterNet2.centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.predictor import VisualizationDemo


# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--save_distance', type=float, default=1.2,
                        help='Threshold to considere an object distance secure respect to the camera')
    parser.add_argument('--nms_max_overlap', type=float, default=0.7,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    parser.add_argument('--max_cosine_distance', type=float, default=0.4,
                        help='Gating threshold for cosine distance metric (object appearance).')
    parser.add_argument('--nn_budget', type=int, default=None,
                        help='Maximum size of the appearance descriptors allery. If None, no budget is enforced.')
    parser.add_argument('--max_iou_distance', type=float, default=0.8,
                        help='maxima iou distance between consecutive tracks')
    parser.add_argument('--max_age', type=int, default=50,
                        help='iterations that a unactive track remains until its deletion')
    parser.add_argument('--n_init', type=int, default=5,
                        help='num of detections of the same object until consider it a track')
    parser.add_argument(
        "--clip_feature_extractor",
        default="ViT-B/16",
        choices=['ViT-B/16', 'ViT-B/32', 'ViT-L/14'],
        help="Clip model to implement the DEEP part of the DeepSORT",
    )
    
    parser.add_argument('--debug', help="Path to debug folder.", default=None)
    parser.add_argument('--desired_fps', type=int, default=0,
                        help='number of frame fps to be proccessed')
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    if args.debug:
        if not os.path.exists(args.debug):
            os.mkdir(args.debug)
        else:
            shutil.rmtree(args.debug)
            os.mkdir(args.debug)

    # Instance Detic Predictor
    try:
        detic_predictor = VisualizationDemo(cfg, args)
    except:
        # second time it works
        detic_predictor = VisualizationDemo(cfg, args)
        logger.warning("w: 'CustomRCNN' was already registered")

    # Instance DeepEstimator Predictor
    absolute_depth = True
    estimator_deep = DeepEstimator(output=args.output, model_path=None, model_type="dpt_hybrid_nyu",
                                   optimize=True, kitti_crop=False, absolute_depth=absolute_depth)

    # Load distance calibration
    d_calibration = np.poly1d(np.load("distance_calibration_function.npy"))

    # initialize deep sort
    nms_max_overlap = args.nms_max_overlap
    max_cosine_distance = args.max_cosine_distance
    nn_budget = args.nn_budget
    model_filename = args.clip_feature_extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = device != "cpu"
    model, transform = clip.load(model_filename, device=device, jit=False)
    model.eval()
    encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)

    # initialize tracker
    tracker = Tracker(metric, max_iou_distance=args.max_iou_distance, max_age=args.max_age, n_init=args.n_init)

    frame_count = 0

    # Infer in single images
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            depth = estimator_deep.estimate(img, path)
            start_time = time.time()
            predictions, visualized_output = detic_predictor.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

    # Infer in stream of frames: WEBCAM
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        # Set input and output videos
        cam = cv2.VideoCapture(int(args.webcam))
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cam.get(cv2.CAP_PROP_FPS)

        # Create Videowriters to generate video output
        file_ext = ".avi"
        path_out_vis = os.path.join(args.output, "webcam_vis" + file_ext)
        path_out_depth = os.path.join(args.output, "webcam_depth" + file_ext)
        output_file_vis = cv2.VideoWriter(path_out_vis, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                          (width, height))
        output_file_depth = cv2.VideoWriter(path_out_depth, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                          (width, height)) # ,0 add last parameter if one chanel output desired

        # Processing loop
        while(cam.isOpened()):
            ret, frame = cam.read()
            # predict depth map
            img_depth = estimator_deep.estimate(frame)
            # predict detections DETIC
            start_time = time.time()
            predictions, visualized_output = detic_predictor.run_on_image(frame)
            bboxes, confs, clss, masks = detic2flat(predictions)  # bboxes in xyxy

            #   when detections are made
            if bboxes.shape[0] != 0:
                #   masked image and depth estimation per detection
                depths, masked_frame = detections_depth(frame, img_depth, masks, bboxes, frame_count, erode_iter=3, debug=args.debug)

                # encode Detic detections and feed to tracker
                bboxes_tlwh = xyxy2tlwh(bboxes)
                bboxes_rd = torch.round(bboxes_tlwh)
                features = encoder(masked_frame, bboxes_rd)  # bboxes format to deepSort -> tlwh
                detections = [Detection(bbox, conf, class_num, feature, depth) for bbox, conf, class_num, feature, depth
                              in zip(
                        bboxes_rd, confs, clss, features, depths)]

                # run non-maxima supression -> tlwh
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                class_nums = np.array([d.class_num for d in detections])
                indices = preprocessing.non_max_suppression(
                    boxs, class_nums, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
            else:
                detections = []

            # Call the tracker and update
            tracker.predict()
            tracker.update(detections)

            # update the tracker outputs: save and plot
            update_tracks(tracker, frame_count, save_txt=False, txt_path="output/detections", save_img=True,
                          view_img=False, img=frame, names=detic_predictor.metadata.thing_classes)

            # write results to video output
            output_file_vis.write(frame)
            if absolute_depth:
                output_file_depth.write(cv2.applyColorMap((img_depth / 257).astype("uint8"), cv2.COLORMAP_JET))  # imwrite only supports uint8
            else:
                output_file_depth.write(img_depth.astype("uint8"))

            frame_count = frame_count + 1
            end_time = time.time() - start_time
            print("Detection and tracking finished in " + str(round(end_time, 2)) + "s")
            
            # Break the loop with 'q' to correctly close videowritter
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                logger.info("Loop break by user")
                break

        # Release VideoCapture and VideoWriters
        output_file_vis.release()
        output_file_depth.release()
        cam.release()
        cv2.destroyAllWindows()

    # Infer in stream of frames: VIDEO
    elif args.video_input:
        # set video input parameters
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        
        # Calculate the number of frames to skip to get the desired fps
        if args.desired_fps == 0:
            desired_fps = original_fps
        else: 
            desired_fps = args.desired_fps
        frame_skip = round(original_fps / desired_fps)

        # Create Videowriters to generate video output
        file_ext = ".avi"
        path_out_vis = os.path.join(args.output, basename.split(".")[0] + file_ext)
        path_out_depth = os.path.join(args.output, basename.split(".")[0] + "_depth" + file_ext)
        output_file_vis = cv2.VideoWriter(path_out_vis, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                          (width, height))
        output_file_depth = cv2.VideoWriter(path_out_depth, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), desired_fps,
                                            (width, height)) # ,0 add last parameter if one chanel output desired

        # Processing loop
        while (video.isOpened()):
            # read frame
            ret, frame = video.read()
            if frame is None:
                break
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            # predict depth map
            img_depth, prediction_depth = estimator_deep.estimate(frame)
            # predict detections DETIC
            start_time = time.time()
            predictions, visualized_output = detic_predictor.run_on_image(frame)
            bboxes, confs, clss, masks = detic2flat(predictions)    # bboxes in xyxy

            #   when detections are made
            if bboxes.shape[0] != 0:
                #   masked image and depth estimation per detection
                depths, masked_frame = detections_depth(frame, img_depth, masks, bboxes, frame_count, erode_iter=4, debug=args.debug)
                real_depths = [correct_distance_calibration(d_calibration, depth) for depth in depths]
                # encode Detic detections and feed to tracker
                bboxes_tlwh = xyxy2tlwh(bboxes)
                bboxes_rd = torch.round(bboxes_tlwh)
                features = encoder(masked_frame, bboxes_rd) # bboxes format to deepSort -> tlwh
                detections = [Detection(bbox, conf, class_num, feature, depth) for bbox, conf, class_num, feature, depth in zip(
                    bboxes_rd, confs, clss, features,  real_depths)]

                # run non-maxima supression -> tlwh
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                class_nums = np.array([d.class_num for d in detections])
                indices = preprocessing.non_max_suppression(
                    boxs, class_nums, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
            else:
                detections = []

            # Call the tracker and update
            tracker.predict()
            tracker.update(detections)

            # update the tracker outputs: save and plot
            update_tracks(tracker, frame_count, save_txt=False, txt_path="output/detections", save_img=True,
                          view_img=False, img=frame, names=detic_predictor.metadata.thing_classes, safe_distance=args.save_distance, print_distance=True)

            # write results to video output
            output_file_vis.write(frame)
            if absolute_depth:
                heatmap = cv2.applyColorMap(((img_depth / 257).astype("uint8")), cv2.COLORMAP_JET)
                output_file_depth.write(heatmap)  # imwrite only supports uint8
            else:
                output_file_depth.write(img_depth.astype("uint8"))

            frame_count = frame_count + 1
            end_time = time.time() - start_time
            print("Detection and tracking finished in " + str(round(end_time, 2)) + "s")

            # Break the loop with 'q' to correctly close videowritter
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                logger.info("Loop break by user")
                break
            
        # Release VideoCapture and VideoWriters
        video.release()
        output_file_vis.release()
        output_file_depth.release()
        try:
            cv2.destroyAllWindows()
        except:
            logger.info("No window to close")
