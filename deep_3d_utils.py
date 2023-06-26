import numpy as np
import torch
import cv2
import random
import os


def detections_depth(frame, img_depth, masks, bboxes, frame_count, erode_iter=1, debug=False):
    dists = []
    mask_base = np.zeros(frame.shape[0:2]).astype("uint8")
    for ii, (mask, bbox) in enumerate(zip(masks, bboxes)):
        mask_np = mask.cpu().numpy().astype("uint8")
        if debug:
            cv2.imwrite(os.path.join(debug, 'f' + str(frame_count) + '_mask_bf_erode' + str(ii) + '.jpg'),
                        mask_np * 255)
        kernel = np.ones((5, 5), np.uint8)
        masked_depth_bf_erode = cv2.bitwise_and(img_depth, img_depth, mask=mask_np)
        mask_erode = cv2.erode(mask_np, kernel, iterations=erode_iter)
        if debug:
            cv2.imwrite(os.path.join(debug, 'f' + str(frame_count) + '_mask_af_erode' + str(ii) + '.jpg'),
                        mask_erode * 255)
        masked_depth = cv2.bitwise_and(img_depth, img_depth, mask=mask_erode)
        if debug:
            heatmap_masked = cv2.applyColorMap(((masked_depth / 257).astype("uint8")), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(debug, 'f' + str(frame_count) + '_mask_depth' + str(ii) + '.jpg'),
                        heatmap_masked)
            erode_results = cv2.bitwise_and(frame, frame, mask=mask_erode)
            cv2.imwrite(os.path.join(debug, 'f' + str(frame_count) + '_erode_fmask' + str(ii) + '.jpg'),
                        erode_results)
        dists.append(mean_detection_depth(masked_depth, np.round(bbox.cpu().numpy()), masked_depth_bf_erode))
        mask_base = cv2.bitwise_or(mask_base * 255, mask_np * 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_base)
    if debug:
        cv2.imwrite(os.path.join(debug, 'f' + str(frame_count) + '_masked_frame' + str(ii) + '.jpg'), masked_frame)
    return dists, masked_frame


def mean_detection_depth(img, bbox, bf_erode):
    xmin, ymin, xmax, ymax = bbox.astype("int")
    if ymax >= img.shape[0]: ymax= img.shape[0] #avoid problems with rounding issues
    if xmax >= img.shape[1]: xmax= img.shape[1] #avoid problems with rounding issues
    sum = 0
    count = 0
    for ii in range(xmin, xmax):
        for jj in range(ymin, ymax):
            if img[jj][ii] != 0:
                sum += img[jj][ii]
                count += 1
    try:
        depth_mean = sum/count
    except:
        # if not pixels with value after erode take the central point of bbox as aproximation
        depth_mean = bf_erode[int(ymax-(ymax-ymin)/2)][int(xmax-(xmax-xmin)/2)]
        
    depth_mean #Normalization
    if depth_mean < 0.5:
        depth_mean = 10000
    return depth_mean


def detic2flat(predictions):
    if "instances" in predictions:
        bboxes = predictions["instances"].pred_boxes.tensor  # boxes in x1y1x2y2 format
        confs = predictions["instances"].scores
        clss = predictions["instances"].pred_classes
        masks = predictions["instances"].pred_masks
        return bboxes, confs, clss, masks
    else:
        return [], [], [], []


def correct_distance_calibration(f_cal, val):
    correct_val = f_cal(val)
    if correct_val > 0:
        return correct_val
    else:
        return 0


#   torch.jit.script
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def xyxy2tlwh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]            # x min
    y[:, 1] = x[:, 1]            # y min
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def update_tracks(tracker, frame_count, save_txt, txt_path, save_img, view_img, img, names, safe_distance=1.2, print_distance=False, thickness=3, info=True):
    if len(tracker.tracks):
        print("[Tracks]", len(tracker.tracks))
        # Obtain the nearest track to the camera
        min_track_dist = 100
        for track in tracker.tracks:
            if track.depth < min_track_dist:
                min_track_dist = track.depth

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        xyxy = track.to_tlbr()
        class_num = track.class_num
        bbox = np.round(xyxy)
        class_name = names[int(class_num)]
        if info:
            print("Tracker ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}, dist: {}".format(
                str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), track.depth))

        if save_txt:  # Write to file  

            # Create folder to store output
            if not os.path.exists(txt_path):
                os.makedirs(txt_path)
                
            xywh = xyxy2xywh(bbox)  # normalized xywh
        
            with open(txt_path + '.txt', 'a') as f:
                f.write('frame: {}; track: {}; class: {}; bbox: {}; dist: {}\n'.format(frame_count, track.track_id, class_num,
                                                                              *xywh), track.depth)

        if save_img or view_img:  # Add bbox to image
            label = f'{class_name} #{track.track_id}'
            if track.depth < safe_distance:
                dist_txt = f' ~ caution'
            else:
                dist_txt = f' ~ safe'
            if track.depth == min_track_dist:
                dist_txt = dist_txt + '*'
            if print_distance:
                dist_txt = dist_txt + f' ~{np.round(track.depth, 1)}m'
            plot_one_box(xyxy, img, label=label+dist_txt,
                         color=get_color_for(label), line_thickness=thickness)


def get_color_for(class_num):
    colors = [
        "#4892EA",
        "#00EEC3",
        "#FE4EF0",
        "#F4004E",
        "#FA7200",
        "#EEEE17",
        "#90FF00",
        "#78C1D2",
        "#8C29FF"
    ]

    num = hash(class_num) # may actually be a number or a string
    hex = colors[num%len(colors)]

    # adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    return rgb

