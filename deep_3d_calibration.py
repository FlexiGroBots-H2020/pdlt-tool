import shutil
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import argparse
import time
import cv2

import sys
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from DPT.run_monodepth_single import DeepEstimator
from deep_3d_utils import detic2flat, xyxy2tlwh, update_tracks, detections_depth
import torch

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools_clip import generate_clip_detections as gdet
import clip



from Detic.third_party.CenterNet2.centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.predictor import VisualizationDemo

def get_parser():
    parser = argparse.ArgumentParser(description="Calibrate depth estimation")
    parser.add_argument("--video-input", help="Videos for calibration.")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
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
    parser.add_argument('--num_samples', type=int, default=2,
                        help='Number of frames used to estimate the average depth of the calibration object')
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--nms_max_overlap', type=float, default=0.7,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    parser.add_argument('--max_cosine_distance', type=float, default=0.4,
                        help='Gating threshold for cosine distance metric (object appearance).')
    parser.add_argument('--nn_budget', type=int, default=None,
                        help='Maximum size of the appearance descriptors allery. If None, no budget is enforced.')

    return parser

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


def get_calibration_func(e_vals, r_vals, save=False, file_name=None):
    z = np.polyfit(e_vals, r_vals, 2)
    p = np.poly1d(z)
    if save:
        np.save(file_name, p)

    return p


def get_measures(args, cfg, video_path, num_samples=10, init_frames=3):
    # set video input parameters
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(video_path)

    # Create Videowriters to generate video output
    file_ext = ".avi"
    path_out_vis = os.path.join(args.output, basename.split(".")[0] + file_ext)
    path_out_depth = os.path.join(args.output, basename.split(".")[0] + "_depth" + file_ext)
    output_file_vis = cv2.VideoWriter(path_out_vis, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                      (width, height))
    output_file_depth = cv2.VideoWriter(path_out_depth, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                        (width, height), 0)

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

    # initialize deep sort
    nms_max_overlap = args.nms_max_overlap
    max_cosine_distance = args.max_cosine_distance
    nn_budget = args.nn_budget
    model_filename = "ViT-B/16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = device != "cpu"
    model, transform = clip.load(model_filename, device=device, jit=False)
    model.eval()
    encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)

    # initialize tracker
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3)

    frame_count = 0
    measures = []
    num_measures = 0

    # Processing loop
    while (num_measures<num_samples):
        # read frame
        ret, frame = video.read()
        if frame is None:
            break
        # predict depth map
        img_depth, predict_depth = estimator_deep.estimate(frame)
        # predict detections DETIC
        start_time = time.time()
        predictions, visualized_output = detic_predictor.run_on_image(frame)
        bboxes, confs, clss, masks = detic2flat(predictions)  # bboxes in xyxy

        #   when detections are made
        if bboxes.shape[0] != 0:
            #   masked image and depth estimation per detection
            depths, masked_frame = detections_depth(frame, img_depth, masks, bboxes, frame_count, erode_iter=3,
                                                    debug="debug")

            # encode Detic detections and feed to tracker
            bboxes_tlwh = xyxy2tlwh(bboxes)
            bboxes_rd = torch.round(bboxes_tlwh)
            features = encoder(masked_frame, bboxes_rd)  # bboxes format to deepSort -> tlwh
            detections = [Detection(bbox, conf, class_num, feature, depth) for bbox, conf, class_num, feature, depth in
                          zip(
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

        if frame_count > init_frames:
            num_measures = num_measures + 1
            measures.append((tracker.tracks[0].depth))

        frame_count = frame_count + 1
        end_time = time.time() - start_time
        print("Detection and tracking finished in " + str(round(end_time, 2)) + "s")

    # Release VideoCapture and VideoWriters
    video.release()
    output_file_vis.release()
    output_file_depth.release()
    try:
        cv2.destroyAllWindows()
    except:
        logger.info("No window to destroy")
    print(np.array(measures))
    return np.round(np.mean(np.array(measures)),2)

if __name__ == '__main__':

    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    measure_depths = []
    real_depths = []

    # Create folders to store outputs
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        shutil.rmtree(args.output)
        os.makedirs(args.output)
    

    for video in os.listdir(args.video_input):
        real_depths.append(int(video.split('.')[0]))
        # obtain original algorithm estimation
        measure_depths.append(get_measures(args, cfg, os.path.join(args.video_input, video), num_samples=args.num_samples))

    # Calibration
    aux = [0]
    estimated_vals = aux + measure_depths
    real_vals = aux + real_depths
    d_calibration_file = 'distance_calibration_function.npy'
    # Get distance calibration function
    d_calibration = get_calibration_func(estimated_vals, real_vals, True, d_calibration_file)

    # Check results
    x = np.arange(5)
    y = d_calibration(x)
    fig = plt.figure()
    plt.plot(x, y,  '-o')

    plt.plot(estimated_vals, real_vals, "o", x, y)
    plt.show()
    fig.savefig('calibration.png')
    print()
