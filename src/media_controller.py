"""
Media Controller Module
Handles system media control commands based on gesture recognition
"""

import os
import time
import platform
import subprocess
from typing import Dict, Optional
import threading
from datetime import datetime

try:
    import pyautogui
    import keyboard
    from pynput.keyboard import Key, Listener
except ImportError:
    print("Warning: Some media control libraries not available")
    pyautogui = None
    keyboard = None

# Windows-specific imports
if platform.system() == "Windows":
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from comtypes import CLSCTX_ALL
        from ctypes import cast, POINTER
        import win32api
        import win32con
        WINDOWS_AVAILABLE = True
    except ImportError:
        print("Warning: Windows-specific libraries not available")
        WINDOWS_AVAILABLE = False
else:
    WINDOWS_AVAILABLE = False

from utils.config import GESTURE_COMMANDS, COMMAND_COOLDOWN


class MediaController:
    """
    Controls system media playback based on gesture commands
    """
    
    def __init__(self):
        self.last_command_time = {}
        self.command_history = []
        self.is_enabled = True
        self.volume_control = None  # Initialize volume_control attribute
        
        # Initialize platform-specific controllers
        self.setup_platform_specific()
        
        # Gesture to action mapping
        self.gesture_actions = {
            'open_palm': self.toggle_play_pause,
            'fist': self.stop,
            'thumbs_up': self.volume_up,
            'thumbs_down': self.volume_down,
            'peace': self.next_track,
            'ok_sign': self.previous_track,
            'pointing': self.toggle_mute
        }
        
        print(f"Media Controller initialized for {platform.system()}")
    
    def setup_platform_specific(self):
        """Setup platform-specific media control"""
        self.platform = platform.system()
        
        # Initialize common attributes
        self.media_keys = {}
        
        if self.platform == "Windows" and WINDOWS_AVAILABLE:
            self.setup_windows()
        elif self.platform == "Darwin":  # macOS
            self.setup_macos()
        elif self.platform == "Linux":
            self.setup_linux()
        else:
            print(f"Platform {self.platform} may have limited media control support")
    
    def setup_windows(self):
        """Setup Windows-specific media control"""
        try:
            # Initialize volume control
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_control = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Windows media keys
            self.media_keys = {
                'play_pause': win32con.VK_MEDIA_PLAY_PAUSE,
                'stop': win32con.VK_MEDIA_STOP,
                'next': win32con.VK_MEDIA_NEXT_TRACK,
                'prev': win32con.VK_MEDIA_PREV_TRACK,
                'volume_up': win32con.VK_VOLUME_UP,
                'volume_down': win32con.VK_VOLUME_DOWN,
                'mute': win32con.VK_VOLUME_MUTE
            }
            
            print("Windows media control initialized")
            
        except Exception as e:
            print(f"Failed to initialize Windows media control: {e}")
            self.volume_control = None
    
    def setup_macos(self):
        """Setup macOS-specific media control"""
        # macOS uses AppleScript for media control
        self.applescript_commands = {
            'play_pause': 'tell application "Music" to playpause',
            'stop': 'tell application "Music" to stop',
            'next': 'tell application "Music" to next track',
            'prev': 'tell application "Music" to previous track',
            'volume_up': 'set volume output volume (output volume of (get volume settings) + 10)',
            'volume_down': 'set volume output volume (output volume of (get volume settings) - 10)',
            'mute': 'set volume with output muted'
        }
        print("macOS media control initialized")
    
    def setup_linux(self):
        """Setup Linux-specific media control"""
        # Linux uses various command-line tools
        self.linux_commands = {
            'play_pause': ['playerctl', 'play-pause'],
            'stop': ['playerctl', 'stop'],
            'next': ['playerctl', 'next'],
            'prev': ['playerctl', 'previous'],
            'volume_up': ['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '+5%'],
            'volume_down': ['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '-5%'],
            'mute': ['pactl', 'set-sink-mute', '@DEFAULT_SINK@', 'toggle']
        }
        print("Linux media control initialized")
    
    def execute_gesture_command(self, gesture: str) -> bool:
        """
        Execute media command based on gesture
        
        Args:
            gesture: Detected gesture name
            
        Returns:
            True if command was executed, False otherwise
        """
        if not self.is_enabled:
            return False
        
        # Check if gesture has a mapped action
        if gesture not in self.gesture_actions:
            return False
        
        # Check cooldown period
        current_time = time.time()
        if gesture in self.last_command_time:
            time_since_last = current_time - self.last_command_time[gesture]
            if time_since_last < COMMAND_COOLDOWN:
                return False
        
        # Execute command
        try:
            action = self.gesture_actions[gesture]
            success = action()
            
            if success:
                self.last_command_time[gesture] = current_time
                self.log_command(gesture)
                print(f"Executed: {gesture} -> {GESTURE_COMMANDS.get(gesture, 'unknown')}")
                return True
                
        except Exception as e:
            print(f"Failed to execute {gesture}: {e}")
        
        return False
    
    def toggle_play_pause(self) -> bool:
        """Toggle play/pause"""
        return self._send_media_key('play_pause')
    
    def stop(self) -> bool:
        """Stop playback"""
        return self._send_media_key('stop')
    
    def next_track(self) -> bool:
        """Skip to next track"""
        return self._send_media_key('next')
    
    def previous_track(self) -> bool:
        """Skip to previous track"""
        return self._send_media_key('prev')
    
    def volume_up(self) -> bool:
        """Increase volume"""
        if self.platform == "Windows" and self.volume_control:
            try:
                current_volume = self.volume_control.GetMasterScalarVolume()
                new_volume = min(1.0, current_volume + 0.1)
                self.volume_control.SetMasterScalarVolume(new_volume, None)
                return True
            except:
                pass
        
        return self._send_media_key('volume_up')
    
    def volume_down(self) -> bool:
        """Decrease volume"""
        if self.platform == "Windows" and self.volume_control:
            try:
                current_volume = self.volume_control.GetMasterScalarVolume()
                new_volume = max(0.0, current_volume - 0.1)
                self.volume_control.SetMasterScalarVolume(new_volume, None)
                return True
            except:
                pass
        
        return self._send_media_key('volume_down')
    
    def toggle_mute(self) -> bool:
        """Toggle mute"""
        if self.platform == "Windows" and self.volume_control:
            try:
                is_muted = self.volume_control.GetMute()
                self.volume_control.SetMute(not is_muted, None)
                return True
            except:
                pass
        
        return self._send_media_key('mute')
    
    def _send_media_key(self, key_name: str) -> bool:
        """
        Send media key command based on platform
        
        Args:
            key_name: Name of the media key to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.platform == "Windows" and WINDOWS_AVAILABLE:
                return self._send_windows_key(key_name)
            elif self.platform == "Darwin":
                return self._send_macos_command(key_name)
            elif self.platform == "Linux":
                return self._send_linux_command(key_name)
            else:
                # Fallback to pyautogui if available
                return self._send_pyautogui_key(key_name)
                
        except Exception as e:
            print(f"Failed to send media key {key_name}: {e}")
            return False
    
    def _send_windows_key(self, key_name: str) -> bool:
        """Send Windows media key"""
        if key_name in self.media_keys:
            try:
                win32api.keybd_event(self.media_keys[key_name], 0, 0, 0)
                win32api.keybd_event(self.media_keys[key_name], 0, win32con.KEYEVENTF_KEYUP, 0)
                return True
            except:
                pass
        return False
    
    def _send_macos_command(self, command_name: str) -> bool:
        """Send macOS AppleScript command"""
        if command_name in self.applescript_commands:
            try:
                subprocess.run(['osascript', '-e', self.applescript_commands[command_name]], 
                             capture_output=True)
                return True
            except:
                pass
        return False
    
    def _send_linux_command(self, command_name: str) -> bool:
        """Send Linux command"""
        if command_name in self.linux_commands:
            try:
                subprocess.run(self.linux_commands[command_name], 
                             capture_output=True, timeout=5)
                return True
            except:
                pass
        return False
    
    def _send_pyautogui_key(self, key_name: str) -> bool:
        """Fallback using pyautogui"""
        if pyautogui is None:
            return False
        
        key_mapping = {
            'play_pause': 'playpause',
            'stop': 'stop',
            'next': 'nexttrack',
            'prev': 'prevtrack',
            'volume_up': 'volumeup',
            'volume_down': 'volumedown',
            'mute': 'volumemute'
        }
        
        if key_name in key_mapping:
            try:
                pyautogui.press(key_mapping[key_name])
                return True
            except:
                pass
        
        return False
    
    def get_volume_level(self) -> Optional[float]:
        """
        Get current volume level (0.0 to 1.0)
        
        Returns:
            Volume level or None if not available
        """
        if self.platform == "Windows" and self.volume_control:
            try:
                return self.volume_control.GetMasterScalarVolume()
            except:
                pass
        
        # For other platforms, would need platform-specific implementation
        return None
    
    def set_volume_level(self, level: float) -> bool:
        """
        Set volume level
        
        Args:
            level: Volume level (0.0 to 1.0)
            
        Returns:
            True if successful
        """
        level = max(0.0, min(1.0, level))  # Clamp to valid range
        
        if self.platform == "Windows" and self.volume_control:
            try:
                self.volume_control.SetMasterScalarVolume(level, None)
                return True
            except:
                pass
        
        return False
    
    def log_command(self, gesture: str):
        """Log executed command"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        command = GESTURE_COMMANDS.get(gesture, 'unknown')
        
        log_entry = {
            'timestamp': timestamp,
            'gesture': gesture,
            'command': command
        }
        
        self.command_history.append(log_entry)
        
        # Keep only last 50 commands
        if len(self.command_history) > 50:
            self.command_history = self.command_history[-50:]
    
    def get_command_history(self) -> list:
        """Get command execution history"""
        return self.command_history.copy()
    
    def enable(self):
        """Enable media control"""
        self.is_enabled = True
        print("Media control enabled")
    
    def disable(self):
        """Disable media control"""
        self.is_enabled = False
        print("Media control disabled")
    
    def test_commands(self):
        """Test all media commands"""
        print("Testing media commands...")
        
        test_gestures = ['open_palm', 'volume_up', 'volume_down', 'next_track']
        
        for gesture in test_gestures:
            print(f"Testing {gesture}...")
            success = self.execute_gesture_command(gesture)
            print(f"  Result: {'Success' if success else 'Failed'}")
            time.sleep(2)  # Wait between commands
        
        print("Test complete!")
    
    def get_status(self) -> dict:
        """Get controller status information"""
        return {
            'enabled': self.is_enabled,
            'platform': self.platform,
            'volume_level': self.get_volume_level(),
            'commands_executed': len(self.command_history),
            'available_gestures': list(self.gesture_actions.keys())
        }


def test_media_controller():
    """Test the media controller"""
    controller = MediaController()
    
    print("Media Controller Test")
    print("Status:", controller.get_status())
    
    # Test individual commands
    print("\nTesting commands...")
    
    # Simulate gesture commands
    test_gestures = ['open_palm', 'thumbs_up', 'thumbs_down']
    
    for gesture in test_gestures:
        print(f"\nTesting gesture: {gesture}")
        success = controller.execute_gesture_command(gesture)
        print(f"Result: {'Success' if success else 'Failed'}")
        
        # Wait to respect cooldown
        time.sleep(COMMAND_COOLDOWN + 0.5)
    
    # Show history
    history = controller.get_command_history()
    if history:
        print("\nCommand History:")
        for entry in history[-5:]:  # Last 5 commands
            print(f"  {entry['timestamp']}: {entry['gesture']} -> {entry['command']}")


if __name__ == "__main__":
    test_media_controller()