# Gesture Media Control 🎵👋

A deep learning-powered application that allows you to control media playback using hand gestures. Built with Python, TensorFlow, OpenCV, and MediaPipe.

![Demo](https://img.shields.io/badge/Status-Working-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- **Real-time Hand Detection**: Uses Google's MediaPipe for accurate hand landmark detection
- **Deep Learning Classification**: Custom neural network trained on hand landmark data
- **Cross-platform Media Control**: Works on Windows, macOS, and Linux
- **Multiple Gestures**: Support for 7 different gesture commands
- **User-friendly GUI**: Complete interface with live camera feed and statistics
- **Custom Training**: Collect your own training data and train custom models
- **High Performance**: Real-time processing at 30+ FPS

## 🖐️ Supported Gestures

| Gesture | Command | Description |
|---------|---------|-------------|
| 🖐️ Open Palm | Play/Pause | Toggle media playback |
| ✊ Fist | Stop | Stop media playback |
| 👍 Thumbs Up | Volume Up | Increase system volume |
| 👎 Thumbs Down | Volume Down | Decrease system volume |
| ✌️ Peace Sign | Next Track | Skip to next track |
| 👌 OK Sign | Previous Track | Go to previous track |
| 👉 Pointing | Mute/Unmute | Toggle system mute |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (built-in or external)
- Windows 10/11, macOS 10.14+, or Linux

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gesture-media-control.git
cd gesture-media-control
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python main.py
```

## 📋 Detailed Setup

### 1. Environment Setup

Create a virtual environment (recommended):
```bash
python -m venv gesture_env
source gesture_env/bin/activate  # On Windows: gesture_env\Scripts\activate
pip install -r requirements.txt
```

### 2. System Requirements

**Hardware:**
- Webcam (720p or higher recommended)
- 8GB RAM minimum
- GPU (optional, but recommended for training)

**Software:**
- Python 3.8+
- Compatible media player (VLC, Windows Media Player, iTunes, etc.)

### 3. Platform-specific Setup

**Windows:**
```bash
pip install pycaw pywin32
```

**macOS:**
```bash
# AppleScript is used (built-in)
```

**Linux:**
```bash
sudo apt-get install playerctl pulseaudio-utils
```

## 🎮 Usage

### First-time Setup

1. **Collect Training Data**
```bash
python main.py --collect
```
- Follow the GUI prompts to collect gesture samples
- Aim for 1000+ samples per gesture for best accuracy
- Ensure good lighting and clear hand visibility

2. **Train the Model**
```bash
python main.py --train
```
- Training typically takes 5-10 minutes
- Model will be saved automatically

3. **Run the Application**
```bash
python main.py
```

### Command Line Options

```bash
python main.py                # Run GUI application (default)
python main.py --collect      # Collect training data
python main.py --train        # Train model
python main.py --test         # Test system components
python main.py --info         # Show system information
```

### GUI Application

The main interface includes:
- **Camera Feed**: Live video with hand detection overlay
- **Control Panel**: Start/stop camera, enable/disable media control
- **Status Panel**: Current gesture, confidence, and statistics
- **Gesture Guide**: Reference for available gestures
- **Command History**: Log of executed commands

## 🏗️ Project Structure

```
gesture_media_control/
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
├── README.md                 # This file
├── src/                      # Source code
│   ├── __init__.py
│   ├── hand_detector.py      # MediaPipe hand detection
│   ├── gesture_classifier.py # Deep learning model
│   ├── media_controller.py   # System media control
│   ├── data_collector.py     # Training data collection
│   └── gui.py               # Main GUI application
├── utils/                    # Utilities
│   ├── __init__.py
│   └── config.py            # Configuration settings
├── data/                     # Training data (created automatically)
│   └── gestures/
└── models/                   # Trained models (created automatically)
    └── gesture_classifier.h5
```

## 🧠 Technical Details

### Architecture

1. **Hand Detection**: MediaPipe extracts 21 hand landmarks in 3D space
2. **Feature Processing**: Landmarks are normalized and smoothed
3. **Gesture Classification**: Neural network classifies gestures from landmarks
4. **Stability Filtering**: Requires consistent detection over multiple frames
5. **Media Control**: Platform-specific commands sent to system

### Neural Network Architecture

```python
Sequential([
    Input(shape=(63,)),           # 21 landmarks × 3 coordinates
    LayerNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='softmax') # 7 gestures + 'none'
])
```

### Performance Optimization

- **Real-time Processing**: 30+ FPS camera processing
- **Gesture Stability**: 5-frame consistency requirement
- **Command Cooldown**: 1-second minimum between identical commands
- **Confidence Threshold**: Configurable detection confidence (default: 0.8)

## 🔧 Configuration

Edit `utils/config.py` to customize:

```python
# Gesture recognition settings
CONFIDENCE_THRESHOLD = 0.8          # Detection confidence
GESTURE_STABILITY_FRAMES = 5        # Frames for stable detection
COMMAND_COOLDOWN = 1.0             # Seconds between commands

# Model settings
HIDDEN_UNITS = [128, 64, 32]       # Neural network architecture
LEARNING_RATE = 0.001              # Training learning rate
EPOCHS = 100                       # Training epochs

# Camera settings
FRAME_WIDTH = 640                  # Camera resolution
FRAME_HEIGHT = 480
FPS = 30                          # Target FPS
```

## 🐛 Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# Test camera access
python -c "import cv2; print('Camera available:' if cv2.VideoCapture(0).isOpened() else 'Camera not found')"
```

**Low gesture recognition accuracy:**
- Ensure good lighting
- Collect more training data
- Retrain the model
- Adjust confidence threshold

**Media commands not working:**
- Check if media player is running
- Verify platform-specific dependencies are installed
- Test with different media applications

**Performance issues:**
- Close unnecessary applications
- Reduce camera resolution in config
- Use GPU acceleration for TensorFlow

### Debug Mode

Run with verbose output:
```bash
python main.py --test  # Test all components
python main.py --info  # Show system information
```

## 📊 Performance Metrics

Typical performance on modern hardware:
- **Detection Accuracy**: 95-99% with proper training
- **Processing Speed**: 30+ FPS real-time
- **Response Time**: <200ms from gesture to action
- **Memory Usage**: ~500MB with TensorFlow
- **Training Time**: 5-10 minutes for 7000 samples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Create a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google MediaPipe** for hand detection technology
- **TensorFlow** for machine learning framework
- **OpenCV** for computer vision utilities
- **tkinter** for GUI framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/gesture-media-control/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gesture-media-control/discussions)
- **Email**: your.email@example.com

## 🔮 Future Enhancements

- [ ] Add more gesture types
- [ ] Implement gesture sequence commands
- [ ] Add voice feedback
- [ ] Create mobile app version
- [ ] Add gesture customization UI
- [ ] Implement real-time training
- [ ] Add multi-hand support
- [ ] Create gesture recording/replay

---

**Made with ❤️ by [Your Name]**

*If you find this project useful, please consider giving it a ⭐ on GitHub!* 