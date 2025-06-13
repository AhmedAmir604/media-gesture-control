"""
Hand Detection Module using MediaPipe
Provides real-time hand landmark detection and preprocessing
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List
from utils.config import MEDIAPIPE_CONFIDENCE, MEDIAPIPE_TRACKING_CONFIDENCE, COLORS


class HandDetector:
    """
    Real-time hand detection and landmark extraction using MediaPipe
    """
    
    def __init__(self, 
                 min_detection_confidence: float = MEDIAPIPE_CONFIDENCE,
                 min_tracking_confidence: float = MEDIAPIPE_TRACKING_CONFIDENCE,
                 max_num_hands: int = 1):
        """
        Initialize MediaPipe hand detection
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            max_num_hands: Maximum number of hands to detect
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmark_history = []
        self.max_history = 10
    
    def detect_hands(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List], np.ndarray]:
        """
        Detect hands and extract landmarks from frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            landmarks: Normalized hand landmarks (21x3 array) or None
            raw_landmarks: Raw MediaPipe landmarks or None
            annotated_frame: Frame with hand annotations
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            # Get first hand (primary hand)
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on frame
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract landmark coordinates
            landmarks = self._extract_landmarks(hand_landmarks, frame.shape)
            
            # Add to history for smoothing
            if landmarks is not None:
                self.landmark_history.append(landmarks)
                if len(self.landmark_history) > self.max_history:
                    self.landmark_history.pop(0)
                
                # Apply smoothing
                smoothed_landmarks = self._smooth_landmarks()
                
                return smoothed_landmarks, hand_landmarks, annotated_frame
        
        return None, None, annotated_frame
    
    def _extract_landmarks(self, hand_landmarks, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Extract and normalize landmark coordinates
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the input frame (height, width, channels)
            
        Returns:
            Normalized landmark array (21x3)
        """
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            # Normalize coordinates relative to frame size
            x = landmark.x
            y = landmark.y
            z = landmark.z  # Relative depth
            
            landmarks.extend([x, y, z])
        
        return np.array(landmarks, dtype=np.float32)
    
    def _smooth_landmarks(self) -> np.ndarray:
        """
        Apply temporal smoothing to landmarks
        
        Returns:
            Smoothed landmark array
        """
        if len(self.landmark_history) == 0:
            return None
        
        # Simple moving average
        landmarks_array = np.array(self.landmark_history)
        return np.mean(landmarks_array, axis=0)
    
    def get_hand_bbox(self, landmarks: np.ndarray, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box around hand
        
        Args:
            landmarks: Hand landmarks array
            frame_shape: Frame dimensions
            
        Returns:
            Bounding box coordinates (x, y, width, height)
        """
        if landmarks is None:
            return None
        
        height, width = frame_shape[:2]
        
        # Extract x, y coordinates
        x_coords = landmarks[0::3] * width
        y_coords = landmarks[1::3] * height
        
        # Calculate bounding box
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        return x_min, y_min, x_max - x_min, y_max - y_min
    
    def draw_info(self, frame: np.ndarray, landmarks: np.ndarray, gesture: str = "", confidence: float = 0.0) -> np.ndarray:
        """
        Draw additional information on frame
        
        Args:
            frame: Input frame
            landmarks: Hand landmarks
            gesture: Detected gesture name
            confidence: Gesture confidence score
            
        Returns:
            Frame with info overlay
        """
        info_frame = frame.copy()
        
        # Draw gesture info
        if gesture:
            text = f"Gesture: {gesture} ({confidence:.2f})"
            cv2.putText(info_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['green'], 2)
        
        # Draw hand detection status
        if landmarks is not None:
            cv2.putText(info_frame, "Hand Detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['green'], 2)
            
            # Draw bounding box
            bbox = self.get_hand_bbox(landmarks, frame.shape)
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(info_frame, (x, y), (x + w, y + h), COLORS['blue'], 2)
        else:
            cv2.putText(info_frame, "No Hand Detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['red'], 2)
        
        return info_frame
    
    def reset_history(self):
        """Reset landmark history"""
        self.landmark_history.clear()
    
    def close(self):
        """Clean up resources"""
        self.hands.close()


def test_hand_detector():
    """Test the hand detector with webcam"""
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    print("Hand Detector Test - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        landmarks, raw_landmarks, annotated_frame = detector.detect_hands(frame)
        
        # Draw additional info
        info_frame = detector.draw_info(annotated_frame, landmarks)
        
        # Display
        cv2.imshow("Hand Detection Test", info_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    test_hand_detector() 