#!/usr/bin/env python3
"""
Gesture Media Control - Complete LIVE Camera Version for Google Colab
Full functionality: live camera, real-time detection, training, and media control
"""

# ==================== IMPORTS & COLAB SETUP ====================
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mediapipe as mp
import os
import time
import pickle
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output, Image, HTML
import threading
import queue
from collections import deque
import base64
from io import BytesIO
from PIL import Image as PILImage
import ipywidgets as widgets

# Google Colab specific imports and setup
try:
    import google.colab
    from google.colab import output
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
    print("🚀 Running in Google Colab!")
    
    # Install required packages
    os.system("pip install mediapipe opencv-python-headless")
    
    # Enable camera access
    from google.colab import drive
    print("📷 Camera access enabled for Colab!")
    
except ImportError:
    IN_COLAB = False
    print("💻 Running locally")

# ==================== COLAB CAMERA INTERFACE ====================
def setup_colab_camera():
    """Setup camera interface for Google Colab"""
    camera_html = """
    <div id="camera-container">
        <video id="camera-feed" width="640" height="480" autoplay></video>
        <canvas id="camera-canvas" width="640" height="480" style="display:none;"></canvas>
        <div>
            <button id="start-camera" onclick="startCamera()">Start Camera</button>
            <button id="stop-camera" onclick="stopCamera()">Stop Camera</button>
            <button id="capture-frame" onclick="captureFrame()">Capture Frame</button>
        </div>
        <div id="status">Camera Status: Ready</div>
    </div>
    
    <script>
    let stream = null;
    let video = document.getElementById('camera-feed');
    let canvas = document.getElementById('camera-canvas');
    let ctx = canvas.getContext('2d');
    
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            video.srcObject = stream;
            document.getElementById('status').innerText = 'Camera Status: Active';
        } catch (err) {
            document.getElementById('status').innerText = 'Camera Status: Error - ' + err.message;
        }
    }
    
    function stopCamera() {
        if (stream) {
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
            document.getElementById('status').innerText = 'Camera Status: Stopped';
        }
    }
    
    function captureFrame() {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
            ctx.drawImage(video, 0, 0, 640, 480);
            let imageData = canvas.toDataURL('image/jpeg');
            google.colab.kernel.invokeFunction('capture_callback', [imageData], {});
        }
    }
    
    // Auto-capture for continuous processing
    function startContinuousCapture() {
        setInterval(() => {
            if (video.videoWidth > 0) {
                captureFrame();
            }
        }, 100); // 10 FPS
    }
    
    // Start continuous capture automatically
    setTimeout(startContinuousCapture, 2000);
    </script>
    """
    
    display(HTML(camera_html))

# Global variable for frame capture
captured_frame = None

def capture_callback(image_data):
    """Callback function for captured frames"""
    global captured_frame
    try:
        # Remove data URL prefix
        image_data = image_data.split(',')[1]
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        # Convert to PIL Image
        pil_image = PILImage.open(BytesIO(image_bytes))
        # Convert to numpy array
        captured_frame = np.array(pil_image)
        # Convert RGB to BGR for OpenCV
        captured_frame = cv2.cvtColor(captured_frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error processing frame: {e}")

# Register the callback
if IN_COLAB:
    output.register_callback('capture_callback', capture_callback)

# ==================== CONFIGURATION ====================
class Config:
    # Gesture classes
    GESTURE_CLASSES = ['open_palm', 'fist', 'thumbs_up', 'thumbs_down', 'peace', 'ok_sign', 'pointing', 'none']
    
    # Data collection settings
    SAMPLES_PER_GESTURE = 300  # Reduced for faster collection in Colab
    COLLECTION_DELAY = 0.1
    
    # Camera settings
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 10  # Reduced for Colab performance
    
    # MediaPipe settings
    MEDIAPIPE_CONFIDENCE = 0.7
    MEDIAPIPE_TRACKING_CONFIDENCE = 0.5
    
    # Model settings
    MODEL_NAME = "gesture_classifier_live.h5"
    SCALER_NAME = "gesture_scaler_live.pkl"
    ENCODER_NAME = "gesture_encoder_live.pkl"
    
    # Gesture command mapping
    GESTURE_COMMANDS = {
        'open_palm': ('play_pause', 'Play/Pause Toggle'),
        'fist': ('stop', 'Stop Playback'),
        'thumbs_up': ('volume_up', 'Volume +10%'),
        'thumbs_down': ('volume_down', 'Volume -10%'),
        'peace': ('next_track', 'Next Track'),
        'ok_sign': ('previous_track', 'Previous Track'),
        'pointing': ('mute_toggle', 'Mute/Unmute')
    }
    
    # Prediction settings
    CONFIDENCE_THRESHOLD = 0.8
    STABILITY_FRAMES = 3
    
    # Colors for visualization
    COLORS = {
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }

# ==================== HAND DETECTOR ====================
class HandDetector:
    """MediaPipe-based hand detection and landmark extraction"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=Config.MEDIAPIPE_CONFIDENCE,
            min_tracking_confidence=Config.MEDIAPIPE_TRACKING_CONFIDENCE
        )
    
    def detect_hands(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List], np.ndarray]:
        """Detect hands and extract landmarks"""
        if frame is None:
            return None, None, np.zeros((480, 640, 3), dtype=np.uint8)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(frame_rgb)
        
        landmarks = None
        hand_landmarks = None
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on frame
            self.mp_draw.draw_landmarks(
                annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )
            
            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks)
        
        return landmarks, hand_landmarks, annotated_frame

# ==================== LIVE DATA COLLECTOR ====================
class LiveDataCollector:
    """Live data collection using Colab camera"""
    
    def __init__(self):
        self.hand_detector = HandDetector()
        self.collected_data = {}
        self.is_collecting = False
        self.current_gesture = None
        self.samples_collected = 0
        self.target_samples = 0
        
        # Create data directory
        os.makedirs("live_gesture_data", exist_ok=True)
        
        # Initialize collected data storage
        for gesture in Config.GESTURE_CLASSES[:-1]:
            self.collected_data[gesture] = []
    
    def start_collection(self, gesture_name: str, num_samples: int = Config.SAMPLES_PER_GESTURE):
        """Start collecting data for a specific gesture"""
        self.current_gesture = gesture_name
        self.target_samples = num_samples
        self.samples_collected = 0
        self.is_collecting = True
        
        print(f"🎯 Starting collection for '{gesture_name}'")
        print(f"📊 Target: {num_samples} samples")
        print("✋ Show the gesture in front of the camera...")
        print("⏸️ Call stop_collection() when done")
    
    def stop_collection(self):
        """Stop data collection"""
        self.is_collecting = False
        if self.current_gesture and self.samples_collected > 0:
            self.save_gesture_data(self.current_gesture, self.collected_data[self.current_gesture])
            print(f"✅ Collection completed! Saved {self.samples_collected} samples for {self.current_gesture}")
        else:
            print("⚠️ No data collected")
    
    def process_frame(self):
        """Process current camera frame for data collection"""
        global captured_frame
        
        if not self.is_collecting or captured_frame is None:
            return False
            
        # Detect hand landmarks
        landmarks, _, annotated_frame = self.hand_detector.detect_hands(captured_frame)
        
        if landmarks is not None and self.samples_collected < self.target_samples:
            # Add to collected data
            self.collected_data[self.current_gesture].append(landmarks.copy())
            self.samples_collected += 1
            
            # Progress update
            if self.samples_collected % 20 == 0:
                print(f"📊 Progress: {self.samples_collected}/{self.target_samples} samples")
            
            # Auto-stop when target reached
            if self.samples_collected >= self.target_samples:
                self.stop_collection()
                return True
                
        return False
    
    def save_gesture_data(self, gesture_name: str, data: List[np.ndarray]):
        """Save collected gesture data"""
        if not data:
            return
        
        df_data = []
        for landmarks in data:
            row = {'gesture': gesture_name}
            for i, coord in enumerate(landmarks):
                row[f'landmark_{i}'] = coord
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        filepath = f"live_gesture_data/{gesture_name}_live.csv"
        df.to_csv(filepath, index=False)
        print(f"💾 Data saved to: {filepath}")
    
    def load_all_data(self) -> pd.DataFrame:
        """Load all collected data"""
        all_data = []
        data_dir = "live_gesture_data"
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.csv'):
                    filepath = os.path.join(data_dir, filename)
                    df = pd.read_csv(filepath)
                    all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"📊 Loaded {len(combined_df)} total samples")
            return combined_df
        else:
            print("⚠️ No data found")
            return pd.DataFrame()
    
    def get_collection_stats(self):
        """Get current collection statistics"""
        stats = {}
        for gesture in Config.GESTURE_CLASSES[:-1]:
            stats[gesture] = len(self.collected_data[gesture])
        
        total = sum(stats.values())
        print(f"📊 Collection Statistics (Total: {total}):")
        for gesture, count in stats.items():
            status = "✅" if count >= Config.SAMPLES_PER_GESTURE else "❌"
            print(f"   {status} {gesture}: {count}/{Config.SAMPLES_PER_GESTURE}")
        
        return stats

# ==================== GESTURE CLASSIFIER ====================
class GestureClassifier:
    """Neural network-based gesture classification"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def create_model(self, input_dim: int, num_classes: int) -> keras.Model:
        """Create optimized neural network for live prediction"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the gesture classification model"""
        print("🚀 Training model for live recognition...")
        
        # Prepare data
        X = data.drop('gesture', axis=1).values
        y = data['gesture'].values
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Create and train model
        self.model = self.create_model(X_scaled.shape[1], len(self.label_encoder.classes_))
        
        # Callbacks for faster training
        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)
        ]
        
        # Train with fewer epochs for faster iteration
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\n📊 Classification Report:")
        print(classification_report(
            y_test, y_pred_classes, 
            target_names=self.label_encoder.classes_
        ))
        
        self.is_trained = True
        
        return {
            'test_accuracy': test_accuracy,
            'history': history,
            'classes': self.label_encoder.classes_
        }
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Fast prediction for live recognition"""
        if not self.is_trained or self.model is None:
            return 'none', 0.0
        
        # Reshape and scale
        landmarks_scaled = self.scaler.transform(landmarks.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(landmarks_scaled, verbose=0)
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)
        
        gesture = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return gesture, confidence
    
    def save_model(self):
        """Save trained model"""
        if self.model:
            self.model.save(Config.MODEL_NAME)
            
            with open(Config.SCALER_NAME, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            with open(Config.ENCODER_NAME, 'wb') as f:
                pickle.dump(self.label_encoder, f)
                
            print("💾 Live model saved!")
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(Config.MODEL_NAME)
            
            with open(Config.SCALER_NAME, 'rb') as f:
                self.scaler = pickle.load(f)
                
            with open(Config.ENCODER_NAME, 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            self.is_trained = True
            print("✅ Live model loaded!")
            return True
        except:
            print("⚠️ No saved model found")
            return False

# ==================== LIVE MEDIA CONTROLLER ====================
class LiveMediaController:
    """Real-time media control with gesture stability"""
    
    def __init__(self):
        self.gesture_commands = Config.GESTURE_COMMANDS
        self.last_command_time = 0
        self.command_cooldown = 2.0  # Longer cooldown for live control
        self.gesture_history = deque(maxlen=Config.STABILITY_FRAMES)
        self.command_count = 0
        
    def process_gesture(self, gesture: str, confidence: float) -> bool:
        """Process gesture with stability checking"""
        current_time = time.time()
        
        # Add to history
        self.gesture_history.append(gesture if confidence > Config.CONFIDENCE_THRESHOLD else 'none')
        
        # Check for stable gesture
        if len(self.gesture_history) == Config.STABILITY_FRAMES:
            stable_gesture = max(set(self.gesture_history), key=list(self.gesture_history).count)
            
            # Execute command if stable and not in cooldown
            if (stable_gesture != 'none' and 
                stable_gesture in self.gesture_commands and
                current_time - self.last_command_time > self.command_cooldown):
                
                return self.execute_command(stable_gesture)
        
        return False
    
    def execute_command(self, gesture: str) -> bool:
        """Execute media command"""
        command, description = self.gesture_commands[gesture]
        self.last_command_time = time.time()
        self.command_count += 1
        
        print(f"🎵 Command #{self.command_count}: {description}")
        self._simulate_command(command)
        
        # Clear history to prevent repeated commands
        self.gesture_history.clear()
        
        return True
    
    def _simulate_command(self, command: str):
        """Simulate media control (replace with actual implementation)"""
        commands = {
            'play_pause': "⏯️ Play/Pause toggled",
            'stop': "⏹️ Playback stopped", 
            'volume_up': "🔊 Volume increased (+10%)",
            'volume_down': "🔉 Volume decreased (-10%)",
            'next_track': "⏭️ Next track",
            'previous_track': "⏮️ Previous track",
            'mute_toggle': "🔇 Mute toggled"
        }
        
        print(f"   → {commands.get(command, 'Unknown command')}")

# ==================== LIVE GESTURE RECOGNITION SYSTEM ====================
class LiveGestureSystem:
    """Complete live gesture recognition and media control system"""
    
    def __init__(self):
        self.hand_detector = HandDetector()
        self.data_collector = LiveDataCollector()
        self.gesture_classifier = GestureClassifier()
        self.media_controller = LiveMediaController()
        
        self.is_running = False
        self.recognition_active = False
        
        print("🎯 Live Gesture Media Control System Ready!")
    
    def collect_all_gestures(self):
        """Interactive collection of all gestures"""
        print("📊 Starting live data collection for all gestures...")
        print("📷 Make sure your camera is active!")
        
        for gesture in Config.GESTURE_CLASSES[:-1]:
            print(f"\n{'='*50}")
            print(f"🎯 Collecting data for: {gesture.upper()}")
            print(f"📋 Get ready to show the '{gesture}' gesture")
            
            input("Press Enter when ready to start collection...")
            
            # Start collection
            self.data_collector.start_collection(gesture, Config.SAMPLES_PER_GESTURE)
            
            # Collection loop
            start_time = time.time()
            while self.data_collector.is_collecting and (time.time() - start_time < 60):  # 60 second timeout
                self.data_collector.process_frame()
                time.sleep(0.1)  # Process at 10 FPS
            
            if self.data_collector.is_collecting:
                self.data_collector.stop_collection()
                print("⏰ Collection timed out")
        
        # Show final statistics
        print(f"\n{'='*50}")
        print("📊 Final Collection Statistics:")
        self.data_collector.get_collection_stats()
    
    def train_live_model(self):
        """Train model on collected data"""
        print("🧠 Training live gesture recognition model...")
        
        # Load collected data
        data = self.data_collector.load_all_data()
        
        if data.empty:
            print("❌ No training data found! Please collect data first.")
            return False
        
        # Train the model
        results = self.gesture_classifier.train(data)
        
        # Save the model
        self.gesture_classifier.save_model()
        
        print(f"\n🎉 Live model training completed!")
        print(f"📊 Test accuracy: {results['test_accuracy']:.4f}")
        
        return True
    
    def start_live_recognition(self):
        """Start live gesture recognition and media control"""
        if not self.gesture_classifier.is_trained:
            if not self.gesture_classifier.load_model():
                print("❌ No trained model! Please train first.")
                return
        
        print("🚀 Starting live gesture recognition...")
        print("📷 Make sure camera is active")
        print("✋ Show gestures to control media")
        print("⏹️ Call stop_live_recognition() to stop")
        
        self.recognition_active = True
        
        # Recognition loop (run this in a separate thread for better performance)
        self._run_recognition_loop()
    
    def stop_live_recognition(self):
        """Stop live recognition"""
        self.recognition_active = False
        print("⏹️ Live recognition stopped")
    
    def _run_recognition_loop(self):
        """Main recognition loop"""
        frame_count = 0
        
        while self.recognition_active:
            global captured_frame
            
            if captured_frame is not None:
                # Detect hand landmarks
                landmarks, _, annotated_frame = self.hand_detector.detect_hands(captured_frame)
                
                if landmarks is not None:
                    # Predict gesture
                    gesture, confidence = self.gesture_classifier.predict(landmarks)
                    
                    # Process gesture for media control
                    command_executed = self.media_controller.process_gesture(gesture, confidence)
                    
                    # Display current prediction (every 10 frames to reduce spam)
                    if frame_count % 10 == 0:
                        status = "🎯" if confidence > Config.CONFIDENCE_THRESHOLD else "⚪"
                        print(f"{status} {gesture} ({confidence:.2f})", end="\r")
                    
                    frame_count += 1
                
            time.sleep(0.1)  # 10 FPS processing
    
    def demo_complete_system(self):
        """Run complete demonstration"""
        print("🚀 COMPLETE LIVE GESTURE MEDIA CONTROL DEMO")
        print("="*60)
        
        # Step 1: Setup camera
        print("\n📷 STEP 1: Camera Setup")
        setup_colab_camera()
        print("✅ Camera interface created - Start your camera above!")
        
        input("\nPress Enter when camera is active...")
        
        # Step 2: Collect data
        print("\n📊 STEP 2: Live Data Collection")
        self.collect_all_gestures()
        
        # Step 3: Train model
        print("\n🧠 STEP 3: Model Training")
        if self.train_live_model():
            
            # Step 4: Live recognition
            print("\n🎯 STEP 4: Live Recognition")
            input("Press Enter to start live gesture recognition...")
            self.start_live_recognition()
            
            print("\n🎉 Live system is now running!")
            print("✅ Show gestures to control media")
            print("📱 Available gestures:")
            for gesture, (_, desc) in Config.GESTURE_COMMANDS.items():
                print(f"   ✋ {gesture} → {desc}")
        
        else:
            print("❌ Training failed!")

# ==================== MAIN INTERFACE ====================
def main():
    """Main function for live Colab system"""
    print("🎯 LIVE Gesture Media Control - Google Colab Version")
    print("="*60)
    
    # Initialize system
    system = LiveGestureSystem()
    
    # Setup camera first
    print("📷 Setting up camera interface...")
    setup_colab_camera()
    
    print("\n🎯 System ready! Available commands:")
    print("1. system.collect_all_gestures() - Collect training data")
    print("2. system.train_live_model() - Train the model")
    print("3. system.start_live_recognition() - Start live control")
    print("4. system.stop_live_recognition() - Stop live control")
    print("5. system.demo_complete_system() - Run full demo")
    
    print("\n📋 Quick start:")
    print("   system.demo_complete_system()")
    
    return system

# ==================== AUTO-RUN ====================
if __name__ == "__main__":
    # Initialize the live system
    live_system = main()
    
    # Auto-run demo if requested
    print("\n🚀 Auto-starting complete demo...")
    live_system.demo_complete_system() 