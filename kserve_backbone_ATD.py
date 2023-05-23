import sys
import kserve
from typing import Dict
import logging
import torch
import time
import json
import os

from kserve_utils import payload2info, ndarray_to_list
import torch

import sys
sys.path.insert(0, 'tph_yolov5/')

from UAV_ATD_kserve import load_model, init_model, infer

#from mqtt_client import MQTTClient
import paho.mqtt.publish as publish

# constants
ENCODING = 'utf-8'

class Model(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        
        logging.info("Init kserve inference service: %s", name)
        
        logging.info("GPU available: %d" , torch.cuda.is_available())
        
        # Define and initialize all needed variables
        try:
            init_model(self)
            logging.info("Model initialized")
        except Exception as e:
            logging.warning("error init model: " + str(e))
        
        # Instance Detic Predictor
        try:
            logging.info("loading model")
            load_model(self)
        except Exception as e:
            logging.warning("error loading model: " + str(e))
        

    def predict(self, request):
        logging.info("Predict call -------------------------------------------------------")
        
        #logging.info("Payload: %s", request)
        start_time = time.time()
        try:
            img, metadata = payload2info(request)
            id = metadata[0]
            frame_id = metadata[1]
            init_time = metadata[2]
            logging.info("Payload image shape: {}, device: {}, frame: {}".format(img.shape, id, frame_id))
        except Exception as e:
            logging.info("Error prepocessing image: {}".format(e))
        
        decode_time = time.time() - start_time
        logging.info(f"Im Decode and metadata extracted in time: {decode_time:.2f}s")
        
        # Predict step
        start_time = time.time()
        try:  
            img_distance, list_tracks, distances, list_dist_awrns = infer(self, img, id, frame_id)
            logging.info(str(list_dist_awrns))
        except Exception as e:
            logging.info("Error processing image: {}".format(e))  
        infer_time = time.time() - start_time
        logging.info(f"Inference time: {infer_time:.2f}s")
        
        # Encode out imgs
        start_time = time.time()
        dict_out = {"init_time": init_time ,
                    "device": id , 
                    "frame": frame_id, 
                    "tracks": list_tracks, 
                    "warnings": list_dist_awrns, 
                    "distances": str(distances)}
        encode_time = time.time() - start_time
        logging.info(f"dict out time: {encode_time:.4f}s")
        logging.info("Image processed")
        
        # Publish a message 
        start_time = time.time()
        mqtt_topic = "common-apps/modtl-model/output/" + id
        client_id = self.name + "_" + id
        publish.single(mqtt_topic, 
                       json.dumps(dict_out), 
                       hostname=os.getenv('BROKER_ADDRESS'), 
                       port=int(os.getenv('BROKER_PORT')), 
                       client_id=client_id, 
                       auth = {"username": os.getenv('BROKER_USER'), "password": os.getenv('BROKER_PASSWORD')} )
        encode_time = time.time() - start_time
        logging.info(f"Publish out time: {encode_time:.2f}s")

        return {}


if __name__ == "__main__":
    model = Model("modtl-model")
    kserve.KFServer().start([model])

