import argparse
import numpy as np
import os
import cv2
import tqdm
import torch
import pickle
import copy

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

import sys
sys.path.append("PerspectiveFields") 

import perspective2d.modeling  # Import custom modeling for perspective estimation
from perspective2d.utils.predictor import VisualizationDemo
from perspective2d.config import get_perspective2d_cfg_defaults
from perspective2d.utils import draw_perspective_fields, draw_from_r_p_f_cx_cy
from perspective2d.utils.panocam import PanoCam
import matplotlib.pyplot as plt

from mplot_thread import Mplot2d
camera_graph1 = Mplot2d(xlabel='X camera', ylabel='Y Camera', title='# XY')

# Choose the model to use
MODEL_ID = 'NEW:Paramnet-360Cities-edina-uncentered'

device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Model configurations and weights for different models
model_zoo = {
    # Configuration for a specific model, including weights and options
    "NEW:Paramnet-360Cities-edina-uncentered": {
        "weights": [
            "https://www.dropbox.com/s/nt29e1pi83mm1va/paramnet_360cities_edina_rpfpp.pth"
        ],
        "opts": [
            "MODEL.WEIGHTS",
            "PerspectiveFields/models/paramnet_360cities_edina_rpfpp.pth",
            "MODEL.DEVICE",
            device,
        ],
        "config_file": "PerspectiveFields/models/paramnet_360cities_edina_rpfpp.yaml",
        "param": True,
    },
    # Additional model configurations can be added here
}

assert MODEL_ID in model_zoo.keys()  # Ensure the selected MODEL_ID exists in the model zoo

def setup_cfg(args):
    """Setup configuration for the model."""
    cfgs = []
    configs = args['config_file'].split('#')
    weights_id = args['opts'].index('MODEL.WEIGHTS') + 1
    weights = args['opts'][weights_id].split('#')
    for i, conf in enumerate(configs):
        if len(conf) != 0:
            tmp_opts = copy.deepcopy(args['opts'])
            tmp_opts[weights_id] = weights[i]
            cfg = get_cfg()
            get_perspective2d_cfg_defaults(cfg)
            cfg.merge_from_file(conf)
            cfg.merge_from_list(tmp_opts)
            cfg.freeze()
            cfgs.append(cfg)
    return cfgs

perspective_cfg_list = setup_cfg(model_zoo[MODEL_ID])  # Setup model configuration

# Lists to store camera parameters for each frame
rolls, pitchs, fovs, cxs, cys, focals = [], [], [], [], [], []

import os

def process_video_frame(frame, frame_number, parameters, demo, camera_graph):
    """Process and analyze a single video frame."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pred = demo.run_on_image(img)

    if 'pred_roll' in pred and 'pred_pitch' in pred and 'pred_general_vfov' in pred:
        parameters['rolls'].append(pred['pred_roll'].cpu().item())
        parameters['pitchs'].append(pred['pred_pitch'].cpu().item())
        parameters['fovs'].append(pred['pred_general_vfov'].cpu().item())

        if 'pred_rel_cx' in pred and 'pred_rel_cy' in pred and 'pred_rel_focal' in pred:
            parameters['cxs'].append(pred['pred_rel_cx'].cpu().item())
            parameters['cys'].append(pred['pred_rel_cy'].cpu().item())
            focal_length = pred['pred_rel_focal'].cpu().item() * frame.shape[0]
            parameters['focals'].append(focal_length)
            camera_graph.draw([frame_number, focal_length], 'Fy', color='b', linewidth=5)

def get_next_exp_folder(base_path="output"):
    exp_folders = [d for d in os.listdir(base_path) if d.startswith("exp") and os.path.isdir(os.path.join(base_path, d))]
    exp_numbers = [int(d.replace("exp", "")) for d in exp_folders]
    next_number = max(exp_numbers) + 1 if exp_numbers else 1
    return os.path.join(base_path, f"exp{next_number}")

def save_averages_and_results(parameters, camera_graph, base_path="output", input_path="input"):
    exp_path = get_next_exp_folder(base_path)
    os.makedirs(exp_path, exist_ok=True)

    with open(os.path.join(exp_path, "averages.txt"), "w") as f:
        f.write(f"File path: {input_path}\n")
        for param, values in parameters.items():
            avg_value = np.mean(values) if values else 0
            f.write(f"Average {param}: {avg_value:.2f}\n")
            print(f"Average {param}: {avg_value:.2f}")

            # Plotting the parameter values and their average
            plt.figure()
            plt.plot(values, label=param)
            plt.axhline(y=avg_value, color='r', linestyle='--', label=f'Avg: {avg_value:.2f}')
            plt.title(f"{param} Evolution and Average")
            plt.xlabel("Time/Frame")
            plt.ylabel(param)
            plt.legend()
            plt.savefig(os.path.join(exp_path, f"{param}_evolution.png"))
            plt.close()


def main(video_path):
    demo = VisualizationDemo(cfg_list=perspective_cfg_list)
    parameters = {'rolls': [], 'pitchs': [], 'fovs': [], 'cxs': [], 'cys': [], 'focals': []}
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video.")
        return

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_video_frame(frame, frame_number, parameters, demo, camera_graph1)
        frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    save_averages_and_results(parameters, camera_graph1, input_path=video_path)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video and extract camera parameters.')
    parser.add_argument('--video_path', type=str, help='Path to the input video file')
    args = parser.parse_args()
    video_path = args.video_path

    main(video_path)
