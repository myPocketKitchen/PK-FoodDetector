# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import utils

import pymongo
import cv2
import matplotlib.pyplot as plt
import numpy as np
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
from datetime import datetime


file = open("srv.txt")
srv = file.read()
file.close()

client = pymongo.MongoClient("{}".format(srv))
food = client.food
records = food.records

in_out = {}

def Average(lst):
    return sum(lst) / len(lst)

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, srv, client, food, records) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=3,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Run object detection estimation using the model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect(rgb_image)
    # Incorporate in-out detection here 
    
    av = []
    data = {}
    item = 0
    box = 0
    
    for x in range(len(detections)):
      if detections[x][1][0][1] > 0.5:
        if "-" in detections[x][1][0][0]: 
          new_item = detections[x][1][0][0]
          box = detections[x][0][3]
          decay_item = new_item.split("-")
          item = decay_item[0]
          status = decay_item[1]
          if status == "-fruit": 
            item = new_item
            status = " "
        else: 
          item = detections[x][1][0][0]
          box = detections[x][0][3]
          status = " "

        if item in in_out:
          if len(in_out[item])<=4:
            av = in_out[item]
            av.insert(0, box)
            in_out.update({item: av})


          elif (in_out[item][0] - in_out[item][4])/5 > 30:
            print(" IN Gradient: ", (in_out[item][0] - in_out[item][4])/5)
            try:
              print("Sent", item)
              data = { 
                'Item' : item,
                'Date Added' :  datetime.today().strftime('%d/%m/%Y'), 
                'Expiry Date' : "N/A", 
                'Status': status
              }
              in_out.pop(item)
              records.insert_one(data)
            except Exception as e:
              print(e)
              pass
          elif (in_out[item][0] - in_out[item][4])/5 < -40:
            print("OUT Gradient: ", (in_out[item][0] - in_out[item][4])/5)
            try:
              print("Removed", item)
              in_out.pop(item)
              records.delete_one({"Food Item" : item})
            except Exception as e:
              print(e)
              pass
          else: 
            av = in_out[item]
            av.pop()
            av.insert(0, box)
            in_out.update({item: av})
        else:
          in_out[item] = [box]

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detections)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU), srv, client, food, records)


if __name__ == '__main__':
  main()
