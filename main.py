#!/usr/bin/env python3


import sys
import os
import argparse
from pathlib import Path

current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from src.gui import GestureMediaControlGUI
    from src.data_collector import DataCollectionGUI, DataCollector
    from src.gesture_classifier import GestureClassifier, train_model
    from src.hand_detector import HandDetector, test_hand_detector
    from src.media_controller import MediaController, test_media_controller
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('tensorflow', 'tensorflow'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('PIL', 'Pillow')
    ]
    
    missing_modules = []
    
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing_modules.append(package_name)
    
    if missing_modules:
        print("Missing required dependencies:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nInstall missing dependencies with:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    return True


def run_gui():
    """Run the main GUI application"""
    print("Starting Gesture Media Control GUI...")
    app = GestureMediaControlGUI()
    app.run()


def run_data_collection():
    """Run data collection interface"""
    print("Starting Data Collection...")
    
    choice = input("Run GUI version? (y/n): ").lower().strip()
    
    if choice == 'y':
        import tkinter as tk
        root = tk.Tk()
        app = DataCollectionGUI()
        app.root = root
        app.setup_gui()
        root.mainloop()
    else:
        collector = DataCollector()
        collector.collect_all_gestures()


def run_training():
    """Train gesture classification model"""
    print("Training gesture classification model...")
    try:
        train_model()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")


def test_components():
    """Test individual components"""
    print("Testing system components...")
    
    print("\n1. Testing Hand Detector...")
    try:
        test_hand_detector()
        print("Hand detector test completed")
    except KeyboardInterrupt:
        print("Hand detector test interrupted")
    except Exception as e:
        print(f"Hand detector test failed: {e}")
    
    print("\n2. Testing Media Controller...")
    try:
        test_media_controller()
        print("Media controller test completed")
    except Exception as e:
        print(f"Media controller test failed: {e}")
    
    print("\n3. Testing Gesture Classifier...")
    try:
        classifier = GestureClassifier()
        if classifier.is_trained:
            print("Gesture classifier loaded successfully")
        else:
            print("No trained model found - please train first")
    except Exception as e:
        print(f"Gesture classifier test failed: {e}")


def show_system_info():
    """Show system information and status"""
    print("=== Gesture Media Control System Info ===\n")
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # Check dependencies
    print("\nDependency status:")
    if check_dependencies():
        print("✓ All dependencies available")
    else:
        print("✗ Missing dependencies (see above)")
        return
    
    # Check data and models
    from utils.config import DATA_DIR, MODELS_DIR, MODEL_PATH
    
    print(f"\nData directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    
    # Check if training data exists
    data_collector = DataCollector()
    stats = data_collector.get_data_statistics()
    print(f"Training samples: {stats['total_samples']}")
    
    if stats['total_samples'] > 0:
        print("Gesture data breakdown:")
        for gesture, count in stats['gestures'].items():
            if count > 0:
                print(f"  {gesture}: {count}")
    
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"✓ Trained model found: {MODEL_PATH}")
    else:
        print("✗ No trained model found")
    
    # Test camera
    print("\nTesting camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera available")
            cap.release()
        else:
            print("✗ Camera not available")
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
    
    # Test media control
    print("\nTesting media control...")
    try:
        controller = MediaController()
        status = controller.get_status()
        print(f"✓ Media controller initialized for {status['platform']}")
    except Exception as e:
        print(f"✗ Media controller failed: {e}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Gesture Media Control - Control media playback with hand gestures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run GUI application
  python main.py --collect          # Collect training data
  python main.py --train            # Train model
  python main.py --test             # Test components
  python main.py --info             # Show system info
        """
    )
    
    parser.add_argument('--collect', action='store_true',
                       help='Run data collection interface')
    parser.add_argument('--train', action='store_true',
                       help='Train gesture classification model')
    parser.add_argument('--test', action='store_true',
                       help='Test system components')
    parser.add_argument('--info', action='store_true',
                       help='Show system information')
    parser.add_argument('--gui', action='store_true',
                       help='Run GUI application (default)')
    
    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    try:
        if args.collect:
            run_data_collection()
        elif args.train:
            run_training()
        elif args.test:
            test_components()
        elif args.info:
            show_system_info()
        else:
            # Default: run GUI
            run_gui()
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 