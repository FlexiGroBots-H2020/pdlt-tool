Situational awareness: people/object detection, tracking and depth estimation with monocular camera

 docker run --name d3d -v $(pwd):/wd --gpus all --network host --device=/dev/video0:/dev/video0 --group-add=video deep3d:v0
 docker run --name d3d -v $(pwd):/wd --gpus all --network host --device=/dev/video0:/dev/video0 --group-add=video -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix deep3d:v0


*video*

docker pull ghcr.io/flexigrobots-h2020/gsaw-tool:v0

docker run -it -v "$(pwd)":/wd/shared --name gsaw ghcr.io/flexigrobots-h2020/gsaw-tool:v0 --yolo-weights yolov8l-seg.pt --source shared/input/VID_20230328_124524.mp4 --tracking-method bytetrack --hide-conf --conf-thres 0.3 --save-vid --project shared/output

*cam*

sudo xhost +
sudo chmod a+rw /dev/video0

docker run -it -e BROKER_PORT=xxxx -e BROKER_PASSWORD="xxxxxx" -e BROKER_USER="xxxx" -e BROKER_ADDRESS="xxxxxx" -v "$(pwd)":/wd/shared --device /dev/video0:/dev/video0 --privileged --net=host -v ~/.Xauthority:/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --name gsaw ghcr.io/flexigrobots-h2020/gsaw-tool:v0 --yolo-weights yolov8l-seg.pt --source 0 --tracking-method bytetrack --hide-conf --conf-thres 0.3 --save-vid --project shared/output --safe-distance 1.5 --save-vid --mqtt_output --mqtt_topic common-apps/modtl-model/output --robot_id tractor_X --fps-stream 5 



*Parameters*

- `--yolo-weights`: The path to the YOLO weights file. YOLO is an object detection model, and this file contains the trained weights for that model.
- `--reid-weights`: The path to the reID weights file. ReID is used for re-identifying objects across different frames in a video.
- `--tracking-method`: The method used for tracking objects. Can be 'strongsort', 'ocsort', or 'bytetrack'.
- `--tracking-config`: The path to the configuration file for tracking.
- `--source`: The source of the input data. Can be a file path, directory path, URL, or a glob pattern. '0' means webcam.
- `--imgsz`: The size of the image(s) used for inference. Height and width can be provided.
- `--conf-thres`: The confidence threshold for detection. Detections with confidence scores below this will be ignored.
- `--iou-thres`: The Intersection Over Union (IOU) threshold used for Non-Maximum Suppression (NMS).
- `--max-det`: The maximum number of detections allowed per image.
- `--device`: The device to use for computations. Can be a specific CUDA device ('0', '0,1,2,3') or 'cpu'.
- `--show-vid`: If set, the tracking video results will be displayed.
- `--save-txt`: If set, the results will be saved to a text file.
- `--save-conf`: If set, the confidence scores will be saved in the text labels.
- `--save-crop`: If set, the cropped prediction boxes will be saved.
- `--save-trajectories`: If set, the trajectories for each track will be saved.
- `--save-vid`: If set, the video tracking results will be saved.
- `--save-json`: If set, the output results will be saved in a JSON file.
- `--nosave`: If set, images/videos will not be saved.
- `--classes`: Filter detections by class. For example, '--classes 0' would only show detections for the class 'person'.
- `--agnostic-nms`: If set, Non-Maximum Suppression (NMS) will be class-agnostic.
- `--augment`: If set, augmented inference will be performed.
- `--visualize`: If set, features will be visualized.
- `--update`: If set, all models will be updated.
- `--project`: The directory where results will be saved.
- `--name`: The name of the experiment. Results will be saved under project/name.
- `--exist-ok`: If set, it's okay if the project/name already exists. It won't increment.
- `--line-thickness`: The thickness of the bounding box lines (in pixels).
- `--hide-labels`: If set, labels will be hidden.
- `--hide-conf`: If set, confidence scores will be hidden.
- `--hide-class`: If set, class IDs will be hidden.
- `--half`: If set, FP16 half-precision inference will be used.
- `--dnn`: If set, OpenCV DNN will be used for ONNX inference.
- `--vid-stride`: The video frame-rate stride.
- `--retina-masks`: If set, masks will be plotted in native resolution.
- `--safe-distance`: The minimum distance to be considered as safe.
- `--mqtt_output`: If set, output will be sent to a MQTT topic.
- `--mqtt_topic`: The name of the output

