import rosbags
from rosbags.rosbag2 import Reader
import cv2
import numpy as np

# Ruta al archivo rosbag
bag_path = 'input/jere/lidarcam_object_det2_0.db3'

# Abrir el archivo rosbag
with Reader(bag_path) as reader:
    for connection, timestamp, rawdata in reader.messages():
        topic = connection.topic
        if topic == '/synced_img':
            # Deserializar imagen y procesar
            msg = reader.read_message(connection, timestamp, rawdata)
            img = ... # Aquí necesitas deserializar la imagen
            cv2.imshow('Image', img)
            cv2.waitKey(1)
        elif topic in ['/synced_pc_lidar_1', '/synced_pc_lidar_2']:
            # Deserializar nube de puntos y procesar
            msg = reader.read_message(connection, timestamp, rawdata)
            pc = ... # Aquí necesitas deserializar la nube de puntos
            # Procesa la nube de puntos según necesites

cv2.destroyAllWindows()
