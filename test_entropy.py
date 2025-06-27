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

    cap = cv2.VideoCapture(int(2))

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:

            # Get frame
            success, frame = cap.read()    
            
            g_success = gaze_detector.detect_gaze(frame)

            if g_success:
                time_ = time.time()
                results = gaze_detector.get_latest_gaze_results()
                print("bboxes:", results.bboxes[0])
                print("pitch:", results.pitch[0])
                print("yaw:", results.yaw[0])
                print("scores:", results.scores[0])

            vframe = gaze_detector.draw_gaze_window()    


            cv2.imshow("Demo",vframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()  