"""
Main GUI Application for Gesture Media Control
Combines hand detection, gesture classification, and media control
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import queue
from typing import Optional

from hand_detector import HandDetector
from gesture_classifier import GestureClassifier
from media_controller import MediaController
from data_collector import DataCollectionGUI
from utils.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, PREVIEW_WIDTH, PREVIEW_HEIGHT,
    GESTURE_CLASSES, GESTURE_COMMANDS, GESTURE_STABILITY_FRAMES,
    CONFIDENCE_THRESHOLD, COLORS
)


class GestureMediaControlGUI:
    """
    Main GUI application for gesture-based media control
    """
    
    def __init__(self):
        # Initialize components
        self.hand_detector = HandDetector()
        self.gesture_classifier = GestureClassifier()
        self.media_controller = MediaController()
        
        # GUI setup
        self.root = tk.Tk()
        self.root.title("Gesture Media Control")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Threading and control variables
        self.camera_thread = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.current_frame = None
        
        # Gesture stability tracking
        self.gesture_history = []
        self.last_stable_gesture = "none"
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'gestures_detected': 0,
            'commands_executed': 0,
            'fps': 0
        }
        
        # Setup GUI components
        self.setup_gui()
        
        # Initialize camera
        self.cap = None
        
    def setup_gui(self):
        """Setup the main GUI interface"""
        # Create main frames
        self.create_menu()
        self.create_control_panel()
        self.create_video_display()
        self.create_status_panel()
        self.create_gesture_info()
        
        # Start GUI update loop
        self.update_gui()
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Settings", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Data menu
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Data", menu=data_menu)
        data_menu.add_command(label="Collect Training Data", command=self.open_data_collection)
        data_menu.add_command(label="Train Model", command=self.train_model)
        data_menu.add_command(label="Load Model", command=self.load_model)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Gesture Guide", command=self.show_gesture_guide)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_control_panel(self):
        """Create control panel"""
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=10, pady=5)
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Camera", 
                                      command=self.toggle_camera)
        self.start_button.grid(row=0, column=0, padx=5)
        
        # Enable/Disable media control
        self.media_enabled_var = tk.BooleanVar(value=True)
        self.media_checkbox = ttk.Checkbutton(control_frame, text="Enable Media Control",
                                            variable=self.media_enabled_var,
                                            command=self.toggle_media_control)
        self.media_checkbox.grid(row=0, column=1, padx=5)
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").grid(row=0, column=2, padx=5)
        self.confidence_var = tk.DoubleVar(value=CONFIDENCE_THRESHOLD)
        confidence_scale = ttk.Scale(control_frame, from_=0.5, to=1.0, 
                                   variable=self.confidence_var, length=150)
        confidence_scale.grid(row=0, column=3, padx=5)
        
        # Settings button
        ttk.Button(control_frame, text="Settings", 
                  command=self.open_settings).grid(row=0, column=4, padx=5)
    
    def create_video_display(self):
        """Create video display area"""
        video_frame = ttk.LabelFrame(self.root, text="Camera Feed", padding="10")
        video_frame.grid(row=1, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                        padx=10, pady=5)
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, width=PREVIEW_WIDTH, 
                                    height=PREVIEW_HEIGHT, bg='black')
        self.video_canvas.grid(row=0, column=0)
        
        # Video controls
        video_controls = ttk.Frame(video_frame)
        video_controls.grid(row=1, column=0, pady=5)
        
        ttk.Button(video_controls, text="Save Frame", 
                  command=self.save_frame).grid(row=0, column=0, padx=2)
        ttk.Button(video_controls, text="Reset", 
                  command=self.reset_detection).grid(row=0, column=1, padx=2)
    
    def create_status_panel(self):
        """Create status and statistics panel"""
        status_frame = ttk.LabelFrame(self.root, text="Status", padding="10")
        status_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N), padx=10, pady=5)
        
        # Status labels
        self.status_vars = {
            'camera': tk.StringVar(value="Camera: Stopped"),
            'gesture': tk.StringVar(value="Gesture: None"),
            'confidence': tk.StringVar(value="Confidence: 0.0"),
            'last_command': tk.StringVar(value="Last Command: None")
        }
        
        for i, (key, var) in enumerate(self.status_vars.items()):
            ttk.Label(status_frame, textvariable=var).grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # Statistics
        stats_frame = ttk.LabelFrame(self.root, text="Statistics", padding="10")
        stats_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        self.stats_vars = {
            'fps': tk.StringVar(value="FPS: 0"),
            'frames': tk.StringVar(value="Frames: 0"),
            'gestures': tk.StringVar(value="Gestures: 0"),
            'commands': tk.StringVar(value="Commands: 0")
        }
        
        for i, (key, var) in enumerate(self.stats_vars.items()):
            ttk.Label(stats_frame, textvariable=var).grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # Reset stats button
        ttk.Button(stats_frame, text="Reset Stats", 
                  command=self.reset_stats).grid(row=len(self.stats_vars), column=0, pady=5)
    
    def create_gesture_info(self):
        """Create gesture information panel"""
        gesture_frame = ttk.LabelFrame(self.root, text="Gesture Commands", padding="10")
        gesture_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        # Create scrollable text widget
        text_frame = ttk.Frame(gesture_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.gesture_text = tk.Text(text_frame, height=10, width=40, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.gesture_text.yview)
        self.gesture_text.configure(yscrollcommand=scrollbar.set)
        
        self.gesture_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Populate gesture info
        self.update_gesture_info()
        
        # Command history
        history_frame = ttk.LabelFrame(self.root, text="Command History", padding="10")
        history_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), 
                          padx=10, pady=5)
        
        self.history_listbox = tk.Listbox(history_frame, height=4)
        self.history_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        history_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, 
                                     command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=history_scroll.set)
        history_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def toggle_camera(self):
        """Start or stop camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera and processing"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            self.is_running = True
            self.start_button.config(text="Stop Camera")
            self.status_vars['camera'].set("Camera: Running")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            print("Camera started successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop camera and processing"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_button.config(text="Start Camera")
        self.status_vars['camera'].set("Camera: Stopped")
        
        # Clear video display
        self.video_canvas.delete("all")
        
        print("Camera stopped")
    
    def camera_loop(self):
        """Main camera processing loop"""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Update statistics
            self.stats['frames_processed'] += 1
            fps_counter += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                self.stats['fps'] = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            # Put frame in queue for GUI display
            if not self.frame_queue.full():
                try:
                    self.frame_queue.put_nowait(processed_frame)
                except queue.Full:
                    pass
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for gesture detection
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed frame with annotations
        """
        # Detect hands
        landmarks, raw_landmarks, annotated_frame = self.hand_detector.detect_hands(frame)
        
        gesture = "none"
        confidence = 0.0
        
        # Classify gesture if hand detected
        if landmarks is not None:
            gesture, confidence = self.gesture_classifier.predict(landmarks)
            
            # Update gesture stability
            self.gesture_history.append(gesture)
            if len(self.gesture_history) > GESTURE_STABILITY_FRAMES:
                self.gesture_history.pop(0)
            
            # Check for stable gesture
            if len(self.gesture_history) == GESTURE_STABILITY_FRAMES:
                if all(g == gesture for g in self.gesture_history):
                    if gesture != self.last_stable_gesture and gesture != "none":
                        # Execute command
                        if self.media_enabled_var.get():
                            success = self.media_controller.execute_gesture_command(gesture)
                            if success:
                                self.stats['commands_executed'] += 1
                                self.update_command_history(gesture)
                        
                        self.last_stable_gesture = gesture
                        self.stats['gestures_detected'] += 1
        else:
            # Reset gesture history if no hand detected
            self.gesture_history.clear()
            self.last_stable_gesture = "none"
        
        # Draw information on frame
        info_frame = self.hand_detector.draw_info(annotated_frame, landmarks, gesture, confidence)
        
        return info_frame
    
    def update_gui(self):
        """Update GUI elements"""
        # Update video display
        if not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                self.display_frame(frame)
                self.current_frame = frame
            except queue.Empty:
                pass
        
        # Update status and statistics
        self.update_status_display()
        
        # Schedule next update
        self.root.after(33, self.update_gui)  # ~30 FPS GUI updates
    
    def display_frame(self, frame: np.ndarray):
        """Display frame in video canvas"""
        # Resize frame to fit canvas
        frame_resized = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # Update canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_image(PREVIEW_WIDTH//2, PREVIEW_HEIGHT//2, 
                                     image=photo)
        
        # Keep a reference to prevent garbage collection
        self.video_canvas.image = photo
    
    def update_status_display(self):
        """Update status variables"""
        # Update statistics
        self.stats_vars['fps'].set(f"FPS: {self.stats['fps']:.1f}")
        self.stats_vars['frames'].set(f"Frames: {self.stats['frames_processed']}")
        self.stats_vars['gestures'].set(f"Gestures: {self.stats['gestures_detected']}")
        self.stats_vars['commands'].set(f"Commands: {self.stats['commands_executed']}")
    
    def update_gesture_info(self):
        """Update gesture information display"""
        self.gesture_text.delete(1.0, tk.END)
        
        info_text = "Available Gestures:\n\n"
        for gesture in GESTURE_CLASSES[:-1]:  # Exclude 'none'
            command = GESTURE_COMMANDS.get(gesture, 'Unknown')
            info_text += f"• {gesture.replace('_', ' ').title()}\n"
            info_text += f"  → {command.replace('_', ' ').title()}\n\n"
        
        info_text += "\nTips:\n"
        info_text += "• Hold gesture steady for recognition\n"
        info_text += "• Ensure good lighting\n"
        info_text += "• Keep hand in camera view\n"
        info_text += "• Adjust confidence threshold if needed"
        
        self.gesture_text.insert(1.0, info_text)
        self.gesture_text.config(state='disabled')
    
    def update_command_history(self, gesture: str):
        """Update command history display"""
        timestamp = time.strftime("%H:%M:%S")
        command = GESTURE_COMMANDS.get(gesture, 'Unknown')
        entry = f"{timestamp} - {gesture} → {command}"
        
        self.history_listbox.insert(0, entry)
        
        # Keep only last 20 entries
        if self.history_listbox.size() > 20:
            self.history_listbox.delete(self.history_listbox.size() - 1)
    
    def toggle_media_control(self):
        """Toggle media control on/off"""
        if self.media_enabled_var.get():
            self.media_controller.enable()
        else:
            self.media_controller.disable()
    
    def reset_detection(self):
        """Reset detection state"""
        self.hand_detector.reset_history()
        self.gesture_history.clear()
        self.last_stable_gesture = "none"
        print("Detection state reset")
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'frames_processed': 0,
            'gestures_detected': 0,
            'commands_executed': 0,
            'fps': 0
        }
        print("Statistics reset")
    
    def save_frame(self):
        """Save current frame to file"""
        if self.current_frame is not None:
            filename = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                cv2.imwrite(filename, self.current_frame)
                messagebox.showinfo("Success", f"Frame saved to {filename}")
    
    def open_data_collection(self):
        """Open data collection window"""
        data_window = tk.Toplevel(self.root)
        app = DataCollectionGUI()
        app.root = data_window
        app.setup_gui()
    
    def train_model(self):
        """Train gesture classification model"""
        def train_thread():
            try:
                self.gesture_classifier.train()
                messagebox.showinfo("Success", "Model training completed!")
            except Exception as e:
                messagebox.showerror("Error", f"Training failed: {e}")
        
        threading.Thread(target=train_thread, daemon=True).start()
        messagebox.showinfo("Training", "Model training started in background...")
    
    def load_model(self):
        """Load trained model"""
        success = self.gesture_classifier.load_model()
        if success:
            messagebox.showinfo("Success", "Model loaded successfully!")
        else:
            messagebox.showerror("Error", "Failed to load model!")
    
    def open_settings(self):
        """Open settings window"""
        # This could be expanded to include more settings
        messagebox.showinfo("Settings", "Settings panel - To be implemented")
    
    def show_gesture_guide(self):
        """Show gesture guide window"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Gesture Guide")
        guide_window.geometry("400x500")
        
        text_widget = tk.Text(guide_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        guide_text = """
Gesture Media Control Guide

AVAILABLE GESTURES:

🖐️ Open Palm
→ Play/Pause media

✊ Fist
→ Stop playback

👍 Thumbs Up
→ Volume up

👎 Thumbs Down
→ Volume down

✌️ Peace Sign
→ Next track

👌 OK Sign
→ Previous track

👉 Pointing
→ Mute/Unmute

TIPS FOR BEST RECOGNITION:

• Ensure good lighting
• Keep hand clearly visible
• Hold gesture steadily for 2-3 seconds
• Position hand in center of camera view
• Avoid background clutter
• Adjust confidence threshold if needed

TROUBLESHOOTING:

• If gestures aren't recognized, try improving lighting
• Make sure the model is trained with your data
• Check that media control is enabled
• Verify your media player is running
"""
        
        text_widget.insert(1.0, guide_text)
        text_widget.config(state='disabled')
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Gesture Media Control v1.0

A deep learning-powered application for controlling media 
playback using hand gestures.

Built with:
• Python 3.8+
• TensorFlow/Keras
• OpenCV
• MediaPipe
• Tkinter

Features:
• Real-time hand detection
• Deep learning gesture classification
• Cross-platform media control
• Custom training data collection
• User-friendly GUI

Developer: [Your Name]
"""
        messagebox.showinfo("About", about_text)
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            self.stop_camera()
        
        # Clean up resources
        if hasattr(self.hand_detector, 'close'):
            self.hand_detector.close()
        
        self.root.destroy()
    
    def run(self):
        """Start the application"""
        print("Starting Gesture Media Control Application...")
        
        # Check if model is available
        if not self.gesture_classifier.is_trained:
            response = messagebox.askyesno(
                "No Model Found", 
                "No trained model found. Would you like to collect training data first?"
            )
            if response:
                self.open_data_collection()
        
        self.root.mainloop()


def main():
    """Main function"""
    try:
        app = GestureMediaControlGUI()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 