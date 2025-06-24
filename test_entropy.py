#!/usr/bin/python3.10

from l2cs.gaze_detectors import Gaze_Detector
import cv2
import time
import torch

if __name__ == '__main__':
    # Initialize the gaze detector
    gaze_detector = Gaze_Detector(
        device='cuda',
        nn_arch='ResNet50',
        weights_pth='/home/vscode/gaze_ws/_L2CSNet_gaze360.pkl'
    )

    cap = cv2.VideoCapture(int(0))

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:

            # Get frame
            success, frame = cap.read()    
            
            g_success = gaze_detector.detect_gaze(frame)
            if not g_success:
                print("Gaze detection failed, retrying...")
                continue
            
            results = gaze_detector

            vframe = gaze_detector.draw_gaze_window()    


            cv2.imshow("Demo",vframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()  