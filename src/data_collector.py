"""
Data Collection Module for Gesture Training
Collects hand landmark data for different gestures
"""

import cv2
import numpy as np
import pandas as pd
import os
import time
from typing import List, Dict
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
import queue

from hand_detector import HandDetector
from utils.config import (
    GESTURE_CLASSES, GESTURES_DIR, SAMPLES_PER_GESTURE, 
    COLLECTION_DELAY, COLORS, FRAME_WIDTH, FRAME_HEIGHT
)


class DataCollector:
    """
    Collects training data for gesture recognition
    """
    
    def __init__(self):
        self.hand_detector = HandDetector()
        self.collected_data = []
        self.current_gesture = None
        self.samples_collected = 0
        self.target_samples = SAMPLES_PER_GESTURE
        self.is_collecting = False
        
        # Ensure gestures directory exists
        for gesture in GESTURE_CLASSES:
            gesture_dir = os.path.join(GESTURES_DIR, gesture)
            os.makedirs(gesture_dir, exist_ok=True)
    
    def collect_gesture_data(self, gesture_name: str, num_samples: int = SAMPLES_PER_GESTURE) -> List[np.ndarray]:
        """
        Collect landmark data for a specific gesture
        
        Args:
            gesture_name: Name of the gesture to collect
            num_samples: Number of samples to collect
            
        Returns:
            List of landmark arrays
        """
        print(f"\nCollecting data for gesture: {gesture_name}")
        print(f"Target samples: {num_samples}")
        print("Position your hand and press SPACE to start collecting")
        print("Press 'q' to quit early")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        gesture_data = []
        samples_collected = 0
        collecting = False
        last_collection_time = 0
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            landmarks, _, annotated_frame = self.hand_detector.detect_hands(frame)
            
            # Draw collection interface
            display_frame = self._draw_collection_interface(
                annotated_frame, gesture_name, samples_collected, num_samples, collecting
            )
            
            # Collect data if conditions are met
            current_time = time.time()
            if (collecting and landmarks is not None and 
                current_time - last_collection_time > COLLECTION_DELAY):
                
                gesture_data.append(landmarks.copy())
                samples_collected += 1
                last_collection_time = current_time
                
                print(f"Collected sample {samples_collected}/{num_samples}")
            
            cv2.imshow(f"Collecting: {gesture_name}", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to start/stop collection
                collecting = not collecting
                if collecting:
                    print("Collection started!")
                else:
                    print("Collection paused!")
            elif key == ord('q'):  # Quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Collection completed! Collected {len(gesture_data)} samples")
        return gesture_data
    
    def _draw_collection_interface(self, frame: np.ndarray, gesture_name: str, 
                                 collected: int, total: int, collecting: bool) -> np.ndarray:
        """
        Draw collection interface on frame
        """
        display_frame = frame.copy()
        
        # Status text
        status = "COLLECTING" if collecting else "PAUSED"
        status_color = COLORS['green'] if collecting else COLORS['red']
        
        cv2.putText(display_frame, f"Gesture: {gesture_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['white'], 2)
        cv2.putText(display_frame, f"Status: {status}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display_frame, f"Progress: {collected}/{total}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['blue'], 2)
        
        # Progress bar
        bar_width = 300
        bar_height = 20
        bar_x, bar_y = 10, 130
        
        # Background
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), COLORS['white'], 2)
        
        # Progress fill
        if total > 0:
            fill_width = int((collected / total) * bar_width)
            cv2.rectangle(display_frame, (bar_x, bar_y), 
                         (bar_x + fill_width, bar_y + bar_height), COLORS['green'], -1)
        
        # Instructions
        cv2.putText(display_frame, "SPACE: Start/Stop | Q: Quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['yellow'], 2)
        
        return display_frame
    
    def save_gesture_data(self, gesture_name: str, data: List[np.ndarray]):
        """
        Save collected gesture data to file
        
        Args:
            gesture_name: Name of the gesture
            data: List of landmark arrays
        """
        if not data:
            print("No data to save!")
            return
        
        # Create DataFrame
        df_data = []
        for landmarks in data:
            row = {'gesture': gesture_name}
            for i, coord in enumerate(landmarks):
                row[f'landmark_{i}'] = coord
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        timestamp = int(time.time())
        filename = f"{gesture_name}_{timestamp}.csv"
        filepath = os.path.join(GESTURES_DIR, gesture_name, filename)
        
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
    
    def collect_all_gestures(self):
        """
        Collect data for all defined gestures
        """
        print("=== Gesture Data Collection ===")
        print(f"Will collect {SAMPLES_PER_GESTURE} samples for each gesture")
        print(f"Gestures to collect: {GESTURE_CLASSES[:-1]}")  # Exclude 'none'
        
        input("Press Enter to start collection...")
        
        # Collect data for each gesture (except 'none')
        for gesture in GESTURE_CLASSES[:-1]:
            print(f"\n--- Starting collection for: {gesture} ---")
            input(f"Get ready to show '{gesture}' gesture. Press Enter when ready...")
            
            data = self.collect_gesture_data(gesture)
            
            if data:
                self.save_gesture_data(gesture, data)
                print(f"✓ Completed: {gesture}")
            else:
                print(f"✗ Failed: {gesture}")
            
            if gesture != GESTURE_CLASSES[-2]:  # If not last gesture
                input("Press Enter to continue to next gesture...")
        
        print("\n=== Collection Complete! ===")
    
    def load_existing_data(self) -> pd.DataFrame:
        """
        Load all existing gesture data
        
        Returns:
            Combined DataFrame with all gesture data
        """
        all_data = []
        
        for gesture in GESTURE_CLASSES:
            gesture_dir = os.path.join(GESTURES_DIR, gesture)
            if os.path.exists(gesture_dir):
                for filename in os.listdir(gesture_dir):
                    if filename.endswith('.csv'):
                        filepath = os.path.join(gesture_dir, filename)
                        df = pd.read_csv(filepath)
                        all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Loaded {len(combined_df)} samples from existing data")
            return combined_df
        else:
            print("No existing data found")
            return pd.DataFrame()
    
    def get_data_statistics(self) -> Dict:
        """
        Get statistics about collected data
        
        Returns:
            Dictionary with data statistics
        """
        df = self.load_existing_data()
        
        if df.empty:
            return {"total_samples": 0, "gestures": {}}
        
        stats = {
            "total_samples": len(df),
            "gestures": {}
        }
        
        for gesture in GESTURE_CLASSES:
            gesture_count = len(df[df['gesture'] == gesture])
            stats["gestures"][gesture] = gesture_count
        
        return stats


class DataCollectionGUI:
    """
    GUI for data collection process
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gesture Data Collection")
        self.root.geometry("600x400")
        
        self.collector = DataCollector()
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Gesture Data Collection", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Current Data Statistics", padding="10")
        stats_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.stats_text = tk.Text(stats_frame, height=8, width=60)
        self.stats_text.grid(row=0, column=0)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        ttk.Button(buttons_frame, text="Refresh Statistics", 
                  command=self.refresh_stats).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(buttons_frame, text="Collect Single Gesture", 
                  command=self.collect_single).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(buttons_frame, text="Collect All Gestures", 
                  command=self.collect_all).grid(row=0, column=2, padx=(0, 10))
        
        # Gesture selection
        gesture_frame = ttk.LabelFrame(main_frame, text="Select Gesture", padding="10")
        gesture_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.gesture_var = tk.StringVar(value=GESTURE_CLASSES[0])
        gesture_combo = ttk.Combobox(gesture_frame, textvariable=self.gesture_var, 
                                   values=GESTURE_CLASSES[:-1], state="readonly")
        gesture_combo.grid(row=0, column=0, padx=(0, 10))
        
        # Initial stats load
        self.refresh_stats()
    
    def refresh_stats(self):
        """Refresh data statistics display"""
        stats = self.collector.get_data_statistics()
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, f"Total Samples: {stats['total_samples']}\n\n")
        
        self.stats_text.insert(tk.END, "Samples per Gesture:\n")
        for gesture, count in stats['gestures'].items():
            self.stats_text.insert(tk.END, f"  {gesture}: {count}\n")
        
        # Highlight gestures that need more data
        self.stats_text.insert(tk.END, "\nStatus:\n")
        for gesture in GESTURE_CLASSES[:-1]:  # Exclude 'none'
            count = stats['gestures'].get(gesture, 0)
            status = "✓" if count >= SAMPLES_PER_GESTURE else "✗"
            self.stats_text.insert(tk.END, f"  {status} {gesture}: {count}/{SAMPLES_PER_GESTURE}\n")
    
    def collect_single(self):
        """Collect data for selected gesture"""
        gesture = self.gesture_var.get()
        if not gesture:
            messagebox.showerror("Error", "Please select a gesture")
            return
        
        # Run collection in separate thread to prevent GUI freeze
        thread = Thread(target=self._run_single_collection, args=(gesture,))
        thread.daemon = True
        thread.start()
    
    def _run_single_collection(self, gesture):
        """Run single gesture collection"""
        try:
            data = self.collector.collect_gesture_data(gesture)
            if data:
                self.collector.save_gesture_data(gesture, data)
                # Refresh stats in main thread
                self.root.after(0, self.refresh_stats)
                messagebox.showinfo("Success", f"Collected {len(data)} samples for {gesture}")
        except Exception as e:
            messagebox.showerror("Error", f"Collection failed: {str(e)}")
    
    def collect_all(self):
        """Collect data for all gestures"""
        result = messagebox.askyesno("Confirm", 
                                   f"This will collect {SAMPLES_PER_GESTURE} samples for each gesture.\n"
                                   "This may take a while. Continue?")
        if result:
            thread = Thread(target=self._run_all_collection)
            thread.daemon = True
            thread.start()
    
    def _run_all_collection(self):
        """Run collection for all gestures"""
        try:
            self.collector.collect_all_gestures()
            self.root.after(0, self.refresh_stats)
            messagebox.showinfo("Success", "All gesture data collected!")
        except Exception as e:
            messagebox.showerror("Error", f"Collection failed: {str(e)}")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


def main():
    """Main function for data collection"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        # Run GUI version
        app = DataCollectionGUI()
        app.run()
    else:
        # Run console version
        collector = DataCollector()
        
        print("Gesture Data Collection")
        print("1. Collect all gestures")
        print("2. Collect single gesture")
        print("3. View statistics")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == "1":
            collector.collect_all_gestures()
        elif choice == "2":
            print("Available gestures:")
            for i, gesture in enumerate(GESTURE_CLASSES[:-1]):
                print(f"{i+1}. {gesture}")
            
            try:
                gesture_idx = int(input("Select gesture number: ")) - 1
                gesture = GESTURE_CLASSES[gesture_idx]
                data = collector.collect_gesture_data(gesture)
                if data:
                    collector.save_gesture_data(gesture, data)
            except (ValueError, IndexError):
                print("Invalid gesture selection")
        elif choice == "3":
            stats = collector.get_data_statistics()
            print(f"\nTotal samples: {stats['total_samples']}")
            for gesture, count in stats['gestures'].items():
                print(f"{gesture}: {count}")


if __name__ == "__main__":
    main() 