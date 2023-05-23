'''
This script implements TPH-yoloV5 with Zero-shot tracker to perform ATD in Drone taken videos or sequence of images 
'''
import sys
import os

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from pathlib import Path
from loguru import logger

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools_clip import generate_clip_detections as gdet
import clip

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'tph-yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'tph_yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from UAV_ATD_utils import parser_ATD, update_tracks, distance_between_bboxes, extract_bbox_from_track, dist_awareness, calculate_font_scale

# Imports TPH-YOLOv5
from tph_yolov5.models.experimental import attempt_load
from tph_yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from tph_yolov5.utils.general import (LOGGER, apply_classifier, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, colorstr, increment_path, non_max_suppression, print_args, save_one_box,
                           scale_coords, strip_optimizer, xyxy2xywh)
from tph_yolov5.utils.plots import Annotator, colors
from tph_yolov5.utils.torch_utils import load_classifier, select_device, time_sync
from tph_yolov5.utils.augmentations import letterbox

import logging
import multiprocessing as mp
import time


class ActiveDevice:
    def __init__(self, device_id, args):
        self.device_id = device_id
        self.request_count = 0
        self.inactive_request_count = 0
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance, args.nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric, args.max_iou_distance, args.max_age, args.n_init) # initialize tracker   

    def increment_request_count(self):
        self.request_count += 1
    
    def increment_inactive_request_count(self):
        self.inactive_request_count += 1

    def __str__(self):
        return f"Device ID: {self.device_id}, Request count: {self.request_count}"


class ActiveDevices:
    def __init__(self):
        self.devices = {}

    def add_device(self, device_id, args):
        if device_id not in self.devices:
            self.devices[device_id] = ActiveDevice(device_id, args)
            logging.debug(f"Added device {device_id}")

    def remove_device(self, device_id):
        if device_id in self.devices:
            del self.devices[device_id]
            logging.debug(f"Removed device {device_id}")

    def get_device(self, device_id):
        if device_id in self.devices:
            return self.devices[device_id]
        else:
            return f"Device ID {device_id} not found."
        
    def get_all_active_devices(self):
        return list(self.devices.values())

    def display_all_devices(self):
        for device in self.devices.values():
            print(device)
            
    def process_request(self, device_id, detections):
        for device in self.devices.values():
            if device.device_id == device_id:
                device.increment_request_count()
                device.tracker.predict()
                device.tracker.update(detections)
            else:
                device.increment_inactive_request_count()

        self.remove_inactive_devices()

    def remove_inactive_devices(self, threshold=100):
        devices_to_remove = []
        for device_id, device in self.devices.items():
            if device.inactive_request_count >= threshold:
                devices_to_remove.append(device_id)

        for device_id in devices_to_remove:
            self.remove_device(device_id)
            logging.debug(f"Removed inactive device {device_id} after {threshold} missed requests")


def init_model(self):
    # Define and initialize all needed variables
    self.encoder = None
    self.model = None
    
    mp.set_start_method("spawn", force=True)
    
    args = parser_ATD()
    self.args = args
    
    # initialize encoder
    model_filename = "models_clip/ViT-B-32.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load(model_filename, device=device, jit=False)
    model.eval()
    self.encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)
    
    # initialize active devices
    self.active_devices = ActiveDevices()
    
    # check HW type
    self.device = select_device(device)
    
   
    
def load_model(self):
    # Load TPH-yolov5 model
    w = str(self.args.weights[0] if isinstance(self.args.weights, list) else self.args.weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    
    self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(self.args.weights, map_location=self.device)
    stride = int(self.model.stride.max())  # model stride
    self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
    
    # Run inference Once
    self.model(torch.zeros(1, 3, *self.args.imgsz).to(self.device).type_as(next(self.model.parameters())))
    logging.info("Model loaded")
    
    
def infer(self, img, id=0, frame_id=0):
    logging.info("Infer step -------------------------------------------------------")
    start_time = time.time()
    self.active_devices.add_device(id, self.args)
    
    s = ""
    logging.info("Infer over: {} {}; from user: {}; and frame_id {}".format(str(type(img)), str(img.dtype), str(id), str(frame_id)))
    logging.info("Device: {}".format(str(self.device)))
    im0 = img.copy() # original dimensions
    logging.info("Original dim: {}".format(str(im0.shape)))
    
    
    # Padded resize
    img = letterbox(img, self.args.imgsz, stride=32, auto=True)[0] # model dimensions
    logging.info("After letterbox dim: {}".format(str(img.shape)))

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    logging.info("After transpose dim: {}".format(str(img.shape)))
    
    img = np.ascontiguousarray(img)
    logging.info("After contiguous dim: {}".format(str(img.shape)))
    
    img = torch.from_numpy(img).to(self.device)    
    logging.info("To tensor done: {}".format(str(type(img))))
    
    img = img.float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    logging.info("Normalized: {}".format(str(type(img))))
    
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    
    logging.info("Infer dim: {}".format(str(img.shape)))
    preprocess_time = time.time() - start_time
    logging.info(f"Preproccess time: {preprocess_time:.4f}s")
    
    
    start_time = time.time()
    # Inference
    pred = self.model(img, augment=False, visualize=False)[0] # buscar parametro no recarga cache
    inference_time = time.time() - start_time
    logging.info(f"Inference time: {inference_time:.4f}s")
    
    start_time = time.time()
    # NMS detected bboxes
    pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres, self.args.classes, self.args.agnostic_nms, max_det=self.args.max_det)
    nms_time = time.time() - start_time
    logging.info(f"NMS time: {nms_time:.4f}s")
    
    # Process predictions
    for i, det in enumerate(pred):  # per image
            
        logging.info("Num detections: {}".format(str(len(det))))
        
        if len(det):
            start_time = time.time()
            # Rescale boxes from img_size to im0 size                
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f'{n} {self.names[int(c)]}s, '  # add to string

            # Transform bboxes from tlbr to tlwh
            trans_bboxes = det[:, :4].clone()
            trans_bboxes[:, 2:] -= trans_bboxes[:, :2]
            bboxes = trans_bboxes[:, :4].cpu()
            confs = det[:, 4]
            class_nums = det[:, -1].cpu()
            classes = class_nums

            # encode yolo detections and feed to tracker
            features = self.encoder(im0, bboxes)
            detections = [Detection(bbox, conf, class_num, feature) for bbox, conf, class_num, feature in zip(
                bboxes, confs, classes, features)]
            
            postproc_time = time.time() - start_time
            logging.info(f"Postproccess det time: {postproc_time:.4f}s")

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            class_nums = np.array([d.class_num for d in detections])
            indices = preprocessing.non_max_suppression(
                boxs, class_nums, self.args.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            start_time = time.time()
            # Call the tracker for this device
            self.active_devices.process_request(id, detections)

            # update tracks
            list_tracks = update_tracks(self.active_devices.get_device(id).tracker, frame_id, self.args.save_txt, self.args.txt_path, self.args.save_img, self.args.show_results, im0, self.names)
            update_tracks_time = time.time() - start_time
            logging.info(f"Update tracks time: {update_tracks_time:.4f}s")
        else:
            logging.info('No detections')
    
    start_time = time.time()
    logging.info('Calculating distances')        
    list_bboxes = extract_bbox_from_track(self.active_devices.get_device(id).tracker)
    if list_bboxes != []:
        distances, img_distance = distance_between_bboxes(list_bboxes, im0, dist_thres=20)
        list_dist_awrns = dist_awareness(self.active_devices.get_device(id).tracker, distances, self.names, dist_awr_thres=10)
        logging.info(list_dist_awrns)

        h,w,c = im0.shape

        offset = 10 

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = calculate_font_scale(w, h)

        for itr, word in enumerate(list_dist_awrns):
            offset += 100
            cv2.putText(im0, word, (20, offset), font, font_size, (0, 0, 255), 3)
                    
    else:
        distances = []
        img_distance = im0
        list_tracks = []
        list_dist_awrns = []
        
    distance_calculation_time = time.time() - start_time
    logging.info(f"Distance calculation time: {distance_calculation_time:.4f}s")

    logging.info(f"Done") 
    
    return img_distance, list_tracks, distances, list_dist_awrns

        