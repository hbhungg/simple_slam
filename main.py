#!/usr/bin/env python3
import numpy as np
import torch
import cv2

from extractor import Extractor


if __name__ == "__main__":
  video_path = "video/test_countryroad.mp4"
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise FileNotFoundError(video_path)

  print(video_path)
  print(f"Frame length: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
  print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
  cv2.namedWindow(video_path)
  cv2.moveWindow(video_path, 1000, 100)

  ex = Extractor()

  # Loop through the video
  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      kps, des, matches = ex.extract_feature(frame)
      if matches is not None:
        # Draw match lines
        for pt1, pt2 in matches:
          cv2.line(frame, tuple(pt1), tuple(pt2), (255,0,0), 2)
      # Draw keypoints
      frame_orb = cv2.drawKeypoints(frame, kps, None, color=(0, 255, 0), flags=0)
      cv2.imshow(video_path, frame_orb)

      if cv2.waitKey(25) & 0xFF == ord("q"): break
    else: break
