#!/usr/bin/python3.10

import os
import pickle
from l2cs.gaze_detectors import Gaze_Detector
import torch
from copy import deepcopy
import math
import pdb
import cv2
import numpy as np
from time import time, sleep, strftime, localtime
from collections import deque
from threading import Thread, Lock

import yaml


CALIBRATION_FILE = '/home/vscode/gaze_ws/calibration_data.pkl'

class AttentionDetector:
    def __init__(self, 
                 attention_threshold=0.5,  # Time in seconds needed to confirm attention
                 pitch_threshold=15,       # Maximum pitch angle for attention
                 yaw_threshold=20,         # Maximum yaw angle for attention
                 history_size=10):         # Number of frames to keep for smoothing
        
        # Initialize the gaze detector
        self.gaze_detector = Gaze_Detector(
            device='cuda',
            nn_arch='ResNet50',
            weights_pth='/home/vscode/gaze_ws/L2CSNet_gaze360.pkl'
        )

        # Initialize parameters
        self.attention_threshold = attention_threshold
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self.attention_start_time = None
        self.attention_state = False
        
        # Initialize angle history for smoothing
        self.angle_history = deque(maxlen=history_size)
        
    
    def smooth_angles(self, angles):
        """Apply smoothing to angles using a moving average"""
        self.angle_history.append(angles)
        return np.mean(self.angle_history, axis=0)
    
    def is_looking_at_robot(self, pitch, yaw):
        """Determine if the person is looking at the robot based on angles"""
        return abs(pitch) < self.pitch_threshold and abs(yaw) < self.yaw_threshold
    
    def process_frame(self, frame):
        # Initialize return values
        attention_detected = False
        sustained_attention = False
        angles = None
        face_found = False

        """Process a single frame and return attention state and visualization"""
        h, w, _ = frame.shape
        
        g_success = self.gaze_detector.detect_gaze(frame)

        if g_success:
            face_found = True
            results = self.gaze_detector.get_latest_gaze_results()
            if results is None:
                return frame, attention_detected, sustained_attention, angles, face_found

            # Extract pitch and yaw from results
            pitch = results.pitch[0]
            yaw = results.yaw[0]

            angles = (math.degrees(pitch), math.degrees(yaw), 0) # Assuming roll is not used and angles are in degrees
            
            # Apply smoothing
            smoothed_angles = self.smooth_angles(angles)
            pitch, yaw, _ = smoothed_angles
            
            # Check if looking at robot
            attention_detected = self.is_looking_at_robot(pitch, yaw)
            
            # Track sustained attention
            current_time = time()
            if attention_detected:
                if self.attention_start_time is None:
                    self.attention_start_time = current_time
                elif (current_time - self.attention_start_time) >= self.attention_threshold:
                    sustained_attention = True
            else:
                self.attention_start_time = None

            frame = self.gaze_detector.draw_gaze_window()

            # Visualization
            color = (0, 255, 0) if sustained_attention else (
                (0, 165, 255) if attention_detected else (0, 0, 255)
            )
            
            # Add text overlays
            cv2.putText(frame, f'Pitch: {int(pitch)}', (20, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f'Yaw: {int(yaw)}', (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw attention status
            status = "Sustained Attention" if sustained_attention else (
                "Attention Detected" if attention_detected else "No Attention"
            )
            cv2.putText(frame, status, (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            
                
        return frame, attention_detected, sustained_attention, angles, face_found

class AttentionCalibrator:
    def __init__(self, 
                 calibration_time=10.0,    # Time in seconds needed for calibration
                 samples_needed=300,        # Number of samples to collect
                 angle_tolerance=15.0):     # Tolerance for angle variation during calibration -- DEFAULT 15.0
        
        self.calibration_time = calibration_time
        self.samples_needed = samples_needed
        self.angle_tolerance = angle_tolerance
        
        # Storage for calibration samples
        self.pitch_samples = []
        self.yaw_samples = []
        
        # Calibration state
        self.calibration_start_time = None
        self.is_calibrated = False
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.pitch_threshold = None
        self.yaw_threshold = None
        
    def start_calibration(self):
        """Start the calibration process"""
        self.calibration_start_time = time()
        self.pitch_samples = []
        self.yaw_samples = []
        self.is_calibrated = False
        print("Starting calibration... Please look directly at the robot.")
        
    def process_calibration_frame(self, pitch, yaw):
        """Process a frame during calibration"""
        if self.calibration_start_time is None:
            return False, "Calibration not started"
        
        current_time = time()
        elapsed_time = current_time - self.calibration_start_time
        
        # Add samples
        self.pitch_samples.append(pitch)
        self.yaw_samples.append(yaw)
        
        # Check if we have enough samples
        if len(self.pitch_samples) >= self.samples_needed:
            # Calculate baseline angles and thresholds
            self.baseline_pitch = np.mean(self.pitch_samples)
            self.baseline_yaw = np.mean(self.yaw_samples)
            
            # Calculate standard deviations
            pitch_std = np.std(self.pitch_samples)
            yaw_std = np.std(self.yaw_samples)
            
            # Set thresholds based on standard deviation and minimum tolerance
            self.pitch_threshold = max(2 * pitch_std, self.angle_tolerance)
            self.yaw_threshold = max(2 * yaw_std, self.angle_tolerance)
            
            self.is_calibrated = True
            return True, "Calibration complete"
        
        # Still calibrating
        return False, f"Calibrating... {len(self.pitch_samples)}/{self.samples_needed} samples"

class CalibratedAttentionDetector(AttentionDetector):
    def __init__(self, calibrator, attention_threshold=0.5, history_size=10):
        super().__init__(
            attention_threshold=attention_threshold,
            pitch_threshold=None,  # Will be set by calibrator
            yaw_threshold=None,    # Will be set by calibrator
            history_size=history_size
        )
        
        # Store calibrator
        self.calibrator = calibrator
        
        # Set thresholds from calibrator
        if calibrator.is_calibrated:
            self.pitch_threshold = calibrator.pitch_threshold
            self.yaw_threshold = calibrator.yaw_threshold
            self.baseline_pitch = calibrator.baseline_pitch
            self.baseline_yaw = calibrator.baseline_yaw
    
    def is_looking_at_robot(self, pitch, yaw):
        """Override the original method to use calibrated values"""
        if not self.calibrator.is_calibrated:
            return False
            
        # Calculate angle differences from baseline
        pitch_diff = abs(pitch - self.calibrator.baseline_pitch)
        yaw_diff = abs(yaw - self.calibrator.baseline_yaw)
        
        return pitch_diff < self.calibrator.pitch_threshold and yaw_diff < self.calibrator.yaw_threshold

def calculate_attention_metrics(attention_window, interval_duration=5.0):
    """
    Calculate attention metrics for a given time window of attention data with improved DGEI.
    
    Logic:
    - Primarily based on robot attention ratio (80% weight)
    - Temporal consistency provides bonus (20% weight)
    - Entropy is used minimally, only as a small modifier
    
    Args:
        attention_window (list): List of tuples (timestamp, attention_state)
        interval_duration (float): Duration of analysis interval in seconds
    
    Returns:
        dict: Dictionary containing attention metrics:
            - attention_ratio: Ratio of frames with attention detected
            - gaze_entropy: Shannon entropy of gaze distribution
            - frames_in_interval: Number of frames in analyzed interval
            - robot_looks: Number of frames looking at robot
            - non_robot_looks: Number of frames not looking at robot
            - gaze_score: Improved DGEI score (0-100)
    """
    
    if not attention_window:
        return {
            'attention_ratio': 0.0,
            'gaze_entropy': 0.0,
            'frames_in_interval': 0,
            'robot_looks': 0,
            'non_robot_looks': 0,
            'gaze_score': 0.0
        }
    
    # Get current time and filter window to only include last interval_duration seconds
    current_time = attention_window[-1][0]  # Latest timestamp
    filtered_window = [(t, a) for t, a in attention_window
                      if current_time - t <= interval_duration]
    
    # Count frames
    frames_in_interval = len(filtered_window)
    robot_looks = sum(1 for _, attention in filtered_window if attention)
    non_robot_looks = frames_in_interval - robot_looks
    
    # Calculate attention ratio (gaze concentration)
    attention_ratio = robot_looks / frames_in_interval if frames_in_interval > 0 else 0.0
    
    # Calculate stationary gaze entropy
    gaze_entropy = 0.0
    if frames_in_interval > 0:
        p_robot = robot_looks / frames_in_interval
        p_non_robot = non_robot_looks / frames_in_interval
        
        # Calculate entropy using Shannon formula
        if p_robot > 0:
            gaze_entropy -= p_robot * math.log2(p_robot)
        if p_non_robot > 0:
            gaze_entropy -= p_non_robot * math.log2(p_non_robot)
    
    # Calculate temporal consistency (how stable the gaze pattern is)
    temporal_consistency = 1.0
    if len(filtered_window) > 1:
        attention_states = [attention for _, attention in filtered_window]
        transitions = 0
        for i in range(1, len(attention_states)):
            if attention_states[i] != attention_states[i-1]:
                transitions += 1
        
        max_transitions = len(attention_states) - 1
        if max_transitions > 0:
            temporal_consistency = 1.0 - (transitions / max_transitions)
    
    ### Simplified DGEI Calculation ###
    
    # If no gaze at robot at all, return 0
    if attention_ratio == 0.0:
        gaze_score = 0.0
    else:
        # Data sufficiency penalty for very small samples
        if frames_in_interval < 3:
            data_sufficiency = frames_in_interval / 3.0
        else:
            data_sufficiency = 1.0
        
        # Primary score: 80% based on robot attention ratio
        primary_score = attention_ratio * 0.8
        
        # Temporal consistency bonus: 20% based on how stable the gaze is
        consistency_bonus = temporal_consistency * 0.2
        
        # Small entropy modifier: slight penalty for very high entropy (>0.8)
        entropy_modifier = 1.0
        if gaze_entropy > 0.8:
            entropy_modifier = 0.95  # 5% penalty for very scattered attention
        
        # Final calculation
        base_score = (primary_score + consistency_bonus) * entropy_modifier
        
        # Apply data sufficiency penalty
        final_score = base_score * data_sufficiency
        
        # Scale to 0-100 and ensure bounds
        gaze_score = final_score * 100
        gaze_score = max(0.0, min(100.0, gaze_score))
    
    ### End DGEI Calculation ###
    
    return {
        'attention_ratio': attention_ratio,
        'gaze_entropy': gaze_entropy,
        'frames_in_interval': frames_in_interval,
        'robot_looks': robot_looks,
        'non_robot_looks': non_robot_looks,
        'gaze_score': gaze_score
    }

class GazeInterfaceController:
    def __init__(self, camera_id=2, gaze_angle_tolerance = 15.0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        self.detector = AttentionDetector()
        self.calibrator = AttentionCalibrator(angle_tolerance=gaze_angle_tolerance)
        self.is_in_attention_detection_mode = False
        self.attention_window_lock = Lock()
        self.gaze_score = 0.0
        self.gaze_score_lock = Lock()
        self.attention_thread = Thread(target=self.attention_detection_loop)
        self.robot_looks_lock = Lock()
        self.robot_looks = 0
        self.gaze_entropy_lock = Lock()
        self.gaze_entropy = 0.0
        self.visualisation_frame = None  # Initialize to prevent AttributeError
                
    def get_gaze_score(self):
        self.gaze_score_lock.acquire()
        score = self.gaze_score
        self.gaze_score_lock.release()
        return score
    
    def get_robot_looks(self):
        self.robot_looks_lock.acquire()
        score = self.robot_looks
        self.robot_looks_lock.release()
        return score
    
    def get_gaze_entropy(self):
        self.gaze_entropy_lock.acquire()
        score = self.gaze_entropy
        self.gaze_entropy_lock.release()
        return score
    
    def get_visualisation_frame(self):
        self.gaze_score_lock.acquire()
        frame = self.visualisation_frame.copy() if self.visualisation_frame is not None else None
        self.gaze_score_lock.release()
        return frame
    
    def kill_attention_thread(self):
        self.is_in_attention_detection_mode = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        sleep(0.5)
        self.attention_thread.join()
        
        
    def calibration_exe(self, no_cal=False):   

        if(no_cal):
            print("Skipping calibration")
            return
             
        # Start calibration
        print("Running Calibration function in Gaze Controller")

        if self.load_calibration_data():  # Try loading existing calibration data
            return 
        
        self.calibrator.start_calibration()
        is_complete = False
        
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
                
            # Process frame using existing detector
            frame, attention, sustained, angles, face_found = self.detector.process_frame(frame)
            
            if face_found and angles is not None:
                pitch, yaw, _ = angles
                is_complete, message = self.calibrator.process_calibration_frame(pitch, yaw)
                
                # Display calibration status
                cv2.putText(frame, message, (20, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                if is_complete:
                    print(f"Calibration complete!")
                    print(f"Baseline Pitch: {self.calibrator.baseline_pitch:.2f}")
                    print(f"Baseline Yaw: {self.calibrator.baseline_yaw:.2f}")
                    print(f"Pitch Threshold: {self.calibrator.pitch_threshold:.2f}")
                    print(f"Yaw Threshold: {self.calibrator.yaw_threshold:.2f}")
                    self.save_calibration_data() 
                    break
            
            self.gaze_score_lock.acquire()
            self.visualisation_frame = frame
            self.gaze_score_lock.release()
            
            cv2.imshow('Calibration', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        if not is_complete:
            # pdb.set_trace()
            print("Calibration interrupted or failed.")
            raise ValueError("Calibration failed")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
    def save_calibration_data(self):
        """Save calibration data to a JSON file."""
        calibration_data = {
            'baseline_pitch': self.calibrator.baseline_pitch,
            'baseline_yaw': self.calibrator.baseline_yaw,
            'pitch_threshold': self.calibrator.pitch_threshold,
            'yaw_threshold': self.calibrator.yaw_threshold
        }

        with open(CALIBRATION_FILE, 'wb') as f:
            pickle.dump(calibration_data, f)
        print("Calibration data saved.")
        
    def load_calibration_data(self):
        """Load existing calibration data if it exists."""
        try:
            with open(CALIBRATION_FILE, 'rb') as f:
                calibration_data = pickle.load(f)
            self.calibrator.baseline_pitch = calibration_data['baseline_pitch']
            self.calibrator.baseline_yaw = calibration_data['baseline_yaw']
            self.calibrator.pitch_threshold = calibration_data['pitch_threshold']
            self.calibrator.yaw_threshold = calibration_data['yaw_threshold']
            self.calibrator.is_calibrated = True
            print("Calibration data loaded successfully.")
            return True
        except FileNotFoundError:
            print("No calibration data found.")
            return False  
        
    def start_detecting_attention(self):
        print("\nStarting attention detection with calibrated values...")
        self.is_in_attention_detection_mode = True
        # Start the attention detection loop in a separate thread
        self.attention_thread.start()
        
        
    def attention_detection_loop(self):
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
        
        # Initialize camera and detector with calibration
        self.detector = CalibratedAttentionDetector(self.calibrator)
        
        self.is_in_attention_detection_mode = True
        
        
        self.attention_window = []
        
        while self.cap.isOpened() and self.is_in_attention_detection_mode:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read frame. Stopping attention detection.")
                break
                
            # print("Processing frame")
            # Process frame
            frame, attention, sustained, angles, face_found = self.detector.process_frame(frame)
            
            # Update attention window
            current_time = time()
            self.attention_window.append((current_time, attention))
            
            # Calculate metrics
            metrics = calculate_attention_metrics(self.attention_window)
            
            self.gaze_score_lock.acquire()
            self.gaze_score = metrics["gaze_score"]
            self.gaze_score_lock.release()
            
            self.robot_looks_lock.acquire()
            self.robot_looks = metrics["robot_looks"]
            self.robot_looks_lock.release()
            
            self.gaze_entropy_lock.acquire()
            self.gaze_entropy = metrics["gaze_entropy"]
            self.gaze_entropy_lock.release()
            
            # Add metrics and calibration values to display
            if face_found:
                h, w, _ = frame.shape
                # Add calibration values
                cv2.putText(frame, f'Baseline Pitch: {self.calibrator.baseline_pitch:.1f}', 
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Baseline Yaw: {self.calibrator.baseline_yaw:.1f}', 
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # Add metrics
                cv2.putText(frame, f'Attention Ratio: {metrics["attention_ratio"]:.2f}', 
                        (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Gaze Entropy: {metrics["gaze_entropy"]:.2f}', 
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Frames in Window: {metrics["frames_in_interval"]}', 
                        (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                self.gaze_score_lock.acquire()
                self.visualisation_frame = frame
                self.gaze_score_lock.release()
                
        self.cap.release()
        cv2.destroyAllWindows()
                
            # Display the frame
            # cv2.imshow('Calibrated HRI Attention Detection', frame)
            
            # Break loop on 'ESC'
            # if cv2.waitKey(5) & 0xFF == 27:
            #     break
        
def save_DGEI_data(name, time_now, training_dict):
    save_path = f'{name}/{name}_{time_now}_DGEI.yaml'
    with open(save_path, 'w') as file:
        yaml.dump(training_dict, file)
    return
        
if __name__=="__main__":
    controller = GazeInterfaceController(camera_id=2)
    controller.calibration_exe()
    controller.start_detecting_attention()
    
    start_time = time()
    duration = 50 * 60  # 5 minutes in seconds
    interval = 3  # Interval in seconds
    next_print_time = start_time + interval
    save_dictionary = {}
    time_step_count = 0
    
    # Format time with current date (DD/MM) and time (HH:MM:SS)
    time_now = strftime('%d-%m_%H-%M-%S', localtime(start_time))

    if not os.path.exists('saved_data'):
        print('we are executing a new session')
        os.makedirs('saved_data')
        print('made directory:' + 'saved_data' + ' for testing data')

    try:
        
        while time() - start_time < duration:            
            # Print the gaze score every 5 seconds
            current_time = time()
            if current_time >= next_print_time:
                gaze_score = controller.get_gaze_score()
                robot_looks = controller.get_robot_looks()
                gaze_entropy = controller.get_gaze_entropy()
                print(f"####### Gaze Score: {gaze_score}")
                print(f"Robot looks: {robot_looks}")
                print(f"Gaze entropy: {gaze_entropy}")
                save_dictionary['gaze_score_timestep_'+str(time_step_count)] = gaze_score
                save_dictionary['robot_looks_timestep_'+str(time_step_count)] = robot_looks
                save_dictionary['gaze_entropy_timestep_'+str(time_step_count)] = gaze_entropy
                time_step_count += 1
                next_print_time = current_time + interval

            save_DGEI_data('saved_data', time_now, save_dictionary)

            frame = controller.get_visualisation_frame()
            if frame is not None:
                f = deepcopy(frame)
                cv2.imshow('Calibrated HRI Attention Detection', f)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            else:
                print("Frame is None")
            sleep(0.05)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        cv2.destroyAllWindows()
        controller.kill_attention_thread()
        exit(0)
    
    print("Attention detection completed.")
    cv2.destroyAllWindows()
    controller.kill_attention_thread()
    exit(0)
    