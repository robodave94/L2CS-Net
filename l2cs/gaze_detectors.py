import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import cv2

from .pipeline import Pipeline
from .vis import render

# create a class to detect faces
class Gaze_Detector:
    def __init__(self, 
                    weights_pth,
                    device='cuda', 
                    nn_arch='ResNet50'):

        if device == 'cuda':
            _device_arg = torch.device('cuda')
        elif device == 'cpu':
            _device_arg = torch.device('cpu')
        else:
            raise ValueError("Device arguement must be 'cuda' or 'cpu'")
        

        self.gaze_pipeline = Pipeline(
        weights=weights_pth,
        arch=nn_arch,
        device = _device_arg
        )

        self._current_gaze_directions = None
        self._last_frame_detected_gaze = False
        self.myFPS = 0.0


    def detect_gaze(self, image):
        start_fps = time.time()  
        self._current_image = image
        # Process the image and return the gaze direction
        try:
            self._current_gaze_directions = self.gaze_pipeline.step(image)
            self._last_frame_detected_gaze = True
        except:
            self._last_frame_detected_gaze = False
        self.myFPS = 1.0 / (time.time() - start_fps)
        
        return self._last_frame_detected_gaze

    def draw_gaze_window(self):
        if self._last_frame_detected_gaze:
            # Draw the gaze direction on the image
            img = render(self._current_image, self._current_gaze_directions)
        else:
            img = self._current_image
        
        fps_img = cv2.putText(img, 'FPS: {:.1f}'.format(self.myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
        
        return fps_img
    
    def get_latest_gaze_results(self):
        if self._last_frame_detected_gaze:
            return self._current_gaze_directions
        else:
            return None
