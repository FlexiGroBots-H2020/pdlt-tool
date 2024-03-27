import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import time
import paho.mqtt.publish as publish
import json

from video_empty import Drainer

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

#OFFSET = 2.2 # to modify the depth estimation depending on light and scene conditions
#OFFSET = 0.55
OFFSET = 1

#Chema parameters
from mplot_thread import Mplot2d
camera_graph1 = Mplot2d(xlabel='X camera', ylabel='Y Camera',title='# XY')
camera_graph2 = Mplot2d(xlabel='X camera', ylabel='Z Camera',title='# XZ')
camera_graph3 = Mplot2d(xlabel='Y camera', ylabel='Z Camera',title='# YZ')
 
from mplot_thread import Mplot3d
#camera_graph_3D = Mplot3d(title='# Camrara Frame 3D')
traj = []
#END Chema

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker


from DepthAnything.run import estimate_depth_relative
from DepthAnything.metric_depth.depth_single_image import Estimator as DA_estimator
from DPT.run_monodepth_single import DeepEstimator
from deep_3d_utils import detections_depth, correct_distance_calibration, replace_inf_with_max
from yolo_utils import scale_masks

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def write_detections_to_file(frame_idx, detections, file_path):
    with open(file_path, 'a') as f:
        f.write(f"{frame_idx} ")
        for detection in detections:
            bbox = detection[:4]
            id = detection[4]
            depth = detection[7]
            f.write(f"{depth:.2f} ")
        f.write("\n")

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.2,  # confidence threshold
        iou_thres=0.3,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        save_json=False,
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        safe_distance=1,
        mqtt_output=False,  
        mqtt_topic="topic",
        robot_id="id",
        fps_stream=10,
        seconds_buffer=20,
        #depth_model="dpt",
        depth_model="depthAnything",
        depth_scenario= "outdoor",
        alpha_weight=0.05, 
        xyz_graphs=True,

):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if (save_txt or save_json) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    
    if depth_model == "dpt":
        # Instance DeepEstimator Predictor
        OFFSET = 2.2
        absolute_depth = True
        estimator_deep = DeepEstimator(output=save_dir, model_path=None, model_type="dpt_hybrid_nyu",
                                    optimize=True, kitti_crop=False, absolute_depth=absolute_depth)

        # Load distance calibration
        d_calibration = np.poly1d(np.load("distance_calibration_function.npy"))
    elif depth_model =="depthAnything":
        # Instance Depth Anything Estimator Predictor
        if depth_scenario == "indoor":
            OFFSET = 1
            path_model = 'local::./DepthAnything/checkpoints/depth_anything_metric_depth_indoor.pt'
        else:
            OFFSET = 0.55
            path_model = 'local::./DepthAnything/checkpoints/depth_anything_metric_depth_outdoor.pt'

        global_settings = { #calculated based in jere Thesis ZED2 / are these parameters been used??
            'FL': 700.819,
            'FY': 771.2,
            'FX': 672.2,
            'NYU_DATA': False,
            'FINAL_HEIGHT': 480,
            'FINAL_WIDTH': 640,
            'DATASET': 'nyu',  
            'pretrained_resource': path_model
        }

        # Uso de la clase Estimator
        estimator_deep_da = DA_estimator('zoedepth', global_settings)
    else:
        logging.error("Depth estimation model not exists")

    # Chema Extrinsic matrix:
    # K = np.array([[700,  0,     imgsz[0]/2],
    #               [0,    700,   imgsz[1]/2],
    #               [0,    0,     1]])
    K = np.array([[global_settings["FX"],  0,     imgsz[0]/2],
                  [0,    global_settings["FY"],   imgsz[1]/2],
                  [0,    0,     1]])

    # Dataloader
    bs = 1
    if webcam:
        #show_vid = check_imshow(warn=True)
        show_vid = False #in Docker not GUI 
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_dp_path, vid_writer, vid_writer_dp, txt_path,  = [None] * bs, [None] * bs, [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs

    # inicializate drainer variables
    drainer = None
    empty_frame = None
    vid_path_drainer = None
    vid_writer_drainer = None

    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch

        # instanciate drainer in first frame
        if drainer == None:
            drainer = Drainer(im.shape[1], im.shape[2])
        # Generate the image processed to display issues
        if im.ndim == 4:
            im = im[0]
        im_yolo = cv2.cvtColor(np.transpose(im, (1, 2, 0)), cv2.COLOR_BGR2RGB)
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

            # predict depth map
            if depth_model == "dpt":
                img_depth = estimator_deep.estimate_gpu(im_yolo)
                tensor = replace_inf_with_max(img_depth)
                img_depth = tensor.detach().cpu().numpy().astype('uint16')  # change to adapt DPT cuda
            else:
                #img_depth_dar = estimate_depth_relative(im_yolo)
                img_depth = estimator_deep_da.infer(im_yolo)
                

        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
     
        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            detections_json = {
                "time": time.time(),
                "frame": dataset.count,
                "detections": []
            }

            if det is not None and len(det):
                if is_seg:
                    shape = im0.shape
                    det_yoloscale = det[:, :4].detach().clone()
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det_yoloscale = det[:, :4]
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # send detections and pass image to image_empty
                drainer.update_empty_image(im_yolo, masks[0], base_alpha=alpha_weight) 
                empty_frame = drainer.get_empty_image()

                # show and save empty image

                cv2.imshow("empty frame", empty_frame)
                cv2.waitKey(1)
                # save final empty_scene
                cv2.imwrite(os.path.join(save_path.replace(save_path.split("/")[-1],"empty_frame.png")), empty_frame)
                if vid_path_drainer == None:  # new video
                    vid_path_drainer = save_path.replace(save_path.split('/')[-1], "drainer_" + save_path.split('/')[-1])
                    if isinstance(vid_writer_drainer, cv2.VideoWriter):
                        vid_writer_drainer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = min(vid_cap.get(cv2.CAP_PROP_FPS), 60)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        w_d, h_d = im_yolo.shape[1], im_yolo.shape[0]
                    else:  # stream
                        fps, w, h = fps_stream, im0.shape[1], im0.shape[0]
                        w_d, h_d = im_yolo.shape[1], im_yolo.shape[0]
                    vid_path_ext = str(Path(vid_path_drainer).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    
                    vid_writer_drainer = cv2.VideoWriter(vid_path_ext, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_d, h_d))
                        
                vid_writer_drainer.write(empty_frame)

                #   masked image and depth estimation per detection
                img_d_rescale = cv2.resize(img_depth,(im_yolo.shape[1], im_yolo.shape[0]))
                depths, masked_frame = detections_depth(im_yolo, img_d_rescale, masks[0], det_yoloscale, frame_idx, erode_iter=1, debug=False) #debug=save_dir

                if depth_model == "dpt":
                    real_depths = [correct_distance_calibration(d_calibration, depth) for depth in depths]
                    real_depths = [x * OFFSET for x in real_depths]
                else:
                    real_depths = [x * OFFSET for x in depths]
                                
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0, real_depths)
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    
                    if is_seg:
                        # Mask plotting
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(np.copy(im0), dtype=torch.float32).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if retina_masks else im[i]
                        )
                    
                    for j, (output) in enumerate(outputs[i]):
                        
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        depth = output[7]

                        if xyz_graphs:
                            #Chema 3D
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            centroidx = output[0] + bbox_w/2
                            centroidy = output[1] + bbox_h/2
                            pp = np.array([centroidx*depth, centroidy*depth, depth])
                            xc, yc, zcc = np.dot(np.linalg.inv(K),pp.T)
                            camera_height = 2
                            # camera to world
                            pitch = np.radians(1)
                            R = np.array([[np.cos(pitch),     0,    np.sin(pitch),    0 ],
                                        [0,                 1,    0,               0 ],
                                        [-np.sin(pitch),    0,    np.cos(pitch),    camera_height],
                                        [ 0,                0,    0,                1 ]])
                            
                            XC = np.array([xc, yc, zcc, 1])
                            xw,yw,zw,zz = np.dot(np.linalg.inv(R),XC.T)

                            #END Chema

                        if save_txt or mqtt_output:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write output results to file
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            if depth < safe_distance:
                                dist_txt = '~caution '
                            else:
                                dist_txt = '~safe '
                            dist_txt += f'{depth:.1f}m'
                            label = None if hide_labels else (f'{id} {names[c]} {dist_txt}' if hide_conf else \
                                (f'{id} {conf:.2f} {dist_txt}' if hide_class else f'{id} {names[c]} {conf:.2f} {dist_txt}'))
                            
                            if save_txt:
                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a+') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                                    
                            detection = {
                                'bbox': str([bbox_left, bbox_top, bbox_w, bbox_h]),  
                                'distance': f'{depth:.1f}m',
                                'distance_awareness': dist_txt,  
                                'msg': label,  
                                'class': names[c], 
                                'class_id': id, 
                                'conf': f'{conf:.2f}',
                            }

                            json_str = json.dumps(detection)
                                
                            detections_json["detections"].append(json_str)

                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            if depth < safe_distance:
                                dist_txt = '~caution '
                            else:
                                dist_txt = '~safe '
                            dist_txt += f'{depth:.1f}m'
                            label = None if hide_labels else (f'{id} {names[c]} {dist_txt}' if hide_conf else \
                                (f'{id} {conf:.2f} {dist_txt}' if hide_class else f'{id} {names[c]} {conf:.2f} {dist_txt}'))
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)

                            #Chema 3d
                            if xyz_graphs:
                                annotator.box_label([centroidx,centroidy,centroidx,centroidy], 'C', color=color)
                                colorsmatplotlib = "bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykwbgrcmykw"
                                camera_graph1.draw([xw ,yw], '# id {}'.format(id), color=colorsmatplotlib[id], linewidth=5)
                                camera_graph2.draw([xw ,zw], '# id {}'.format(id), color=colorsmatplotlib[id], linewidth=5)
                                camera_graph3.draw([yw ,zw], '# id {}'.format(id), color=colorsmatplotlib[id], linewidth=5)
                                traj.append([xc ,yc, zcc])
                                #camera_graph_3D.drawTraj(traj, '# id {}'.format(id), color=colorsmatplotlib[id])
                                #End Chema

                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(np.array(bbox, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                            
                            if xyz_graphs:
                                #Chema 3d
                                camera_graph1.refresh()
                                camera_graph2.refresh()
                                camera_graph3.refresh()
                                #camera_graph_3D.refresh()
                                #End Chema
            else:   
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()
            im0 = annotator.result()

            text_location = (10, 25)  # Puedes ajustar estos valores según sea necesario
            font = cv2.FONT_HERSHEY_SIMPLEX  # O el tipo de fuente que prefieras
            font_scale = 0.5  # Tamaño de la fuente
            color = (255, 255, 255)  # Color del texto en BGR (aquí, blanco)
            thickness = 2  # Espesor de la línea del texto
            frame_id_text = f'{frame_idx}'

            # Colocamos el texto en la imagen
            cv2.putText(im0, frame_id_text, text_location, font, font_scale, color, thickness, cv2.LINE_AA)

            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if webcam:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path  
                        vid_dp_path[i] = save_path.replace(save_path.split('/')[-1], "dpt_" + save_path.split('/')[-1])
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                            vid_writer_dp[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = min(vid_cap.get(cv2.CAP_PROP_FPS), 60)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w_d, h_d = im_yolo.shape[1], im_yolo.shape[0]
                        else:  # stream
                            fps, w, h = fps_stream, im0.shape[1], im0.shape[0]
                            w_d, h_d = im_yolo.shape[1], im_yolo.shape[0]
                        vid_path_ext = str(Path(vid_path[i]).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_dp_path_ext = str(Path(vid_dp_path[i]).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(vid_path_ext, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer_dp[i] = cv2.VideoWriter(vid_dp_path_ext, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_d, h_d))
                    vid_writer[i].write(im0)
                    if depth_model == "dpt":
                        heatmap = cv2.applyColorMap(((img_depth / 257).astype("uint8")), cv2.COLORMAP_JET)
                    else:
                        heatmap = cv2.applyColorMap((img_depth).astype("uint8"), cv2.COLORMAP_JET)
                    vid_writer_dp[i].write(heatmap)  # imwrite only supports uint8
                    # save actual videos and start other
                    if (frame_idx % (fps_stream * seconds_buffer)) == 0:
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                            vid_writer_dp[i].release()  # release previous video writer
                        save_path_vid = vid_path[i] + f"_buffered.mp4"
                        save_path_vid_dp = vid_dp_path[i] + f"_buffered.mp4"
                        os.rename(vid_path_ext, save_path_vid)
                        os.rename(vid_dp_path_ext, save_path_vid_dp)
                        logging.info("Video buffered")
                        # reopen the live ones
                        vid_writer[i] = cv2.VideoWriter(vid_path_ext, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer_dp[i] = cv2.VideoWriter(vid_dp_path_ext, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_d, h_d))
                else:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path  
                        vid_dp_path[i] = save_path.replace(save_path.split('/')[-1], "dpt_" + save_path.split('/')[-1])
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                            vid_writer_dp[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = min(vid_cap.get(cv2.CAP_PROP_FPS), 60)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w_d, h_d = im_yolo.shape[1], im_yolo.shape[0]
                        else:  # stream
                            fps, w, h = fps_stream, im0.shape[1], im0.shape[0]
                            w_d, h_d = im_yolo.shape[1], im_yolo.shape[0]
                        vid_path_ext = str(Path(vid_path[i]).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_dp_path_ext = str(Path(vid_dp_path[i]).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(vid_path_ext, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer_dp[i] = cv2.VideoWriter(vid_dp_path_ext, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_d, h_d))
                    vid_writer[i].write(im0)
                    if depth_model == "dpt":
                        heatmap = cv2.applyColorMap(((img_depth / 257).astype("uint8")), cv2.COLORMAP_JET)
                    else:
                        heatmap = cv2.applyColorMap((img_depth).astype("uint8"), cv2.COLORMAP_JET)
                    vid_writer_dp[i].write(heatmap)  # imwrite only supports uint8


            if frame_idx % 10 == 0:  # Cada 10 frames
                # Define el nombre del archivo basado en el nombre del video de salida
                output_file_path = str(save_dir / f"{txt_file_name}_distances.txt")
                write_detections_to_file(frame_idx, outputs[i], output_file_path)        
            
            if mqtt_output:
                # publish only 1 msg per second
                if (frame_idx % (fps_stream)) == 0:
                    # Publish a message 
                    start_time = time.time()
                    mqtt_topic_publish = os.path.join(mqtt_topic, robot_id)
                    client_id = robot_id + str(source)
                    dict_out = json.dumps(detections_json)
                    publish.single(mqtt_topic_publish, 
                                json.dumps(dict_out), 
                                hostname=os.getenv('BROKER_ADDRESS'), 
                                port=int(os.getenv('BROKER_PORT')), 
                                client_id=client_id, 
                                auth = {"username": os.getenv('BROKER_USER'), "password": os.getenv('BROKER_PASSWORD')} )
                    encode_time = time.time() - start_time
                    logging.info(f"Publish out time: {encode_time:.2f}s")
                    if save_json:
                        with open(txt_path + '.json', 'a+') as f:
                            f.write(dict_out)


            prev_frames[i] = curr_frames[i]
            
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")


    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)

    vid_writer_drainer.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8s-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-json', action='store_true', help='save json output results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--safe-distance', type=float, default=1, help='min distance to be considered as safe')
    parser.add_argument('--mqtt_output', default=False, action='store_true', help='send output to mqtt topic')
    parser.add_argument('--mqtt_topic', default='common-apps/modtl-model/output', help='output topic name')
    parser.add_argument('--robot_id', default='robot_A', help='device id')
    parser.add_argument('--fps-stream', type=int, default=5, help='frames per second of the streaming')
    parser.add_argument('--seconds-buffer', type=int, default=30, help='video seconds buffered')
    parser.add_argument('--depth_model', default="depthAnything", help= 'select if want dpt or Depht-Anything model')
    parser.add_argument('--depth_scenario', default="outdoor", help= 'Select indoor or outdoor model in Depht-Anything')
    parser.add_argument('--xyz_graphs', default=True, action='store_false', help='print xyz track evolution graphs')
    parser.add_argument('--alpha_weight', type=float, default=0.02, help='how much modify the empty image each iteration')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
