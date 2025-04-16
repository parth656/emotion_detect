import cv2
import random
import os
import time
import numpy as np
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QStatusBar, 
                            QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor


class SimpleEmotionDetection:
    def __init__(self):
        # Emotions list
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.current_emotion = "neutral"
        
        # Camera settings
        self.width = 640
        self.height = 480
        self.cap = None
        
        # Find face cascade file
        self.face_cascade = self.load_face_cascade()
        
        # Initialize recommendations
        self.init_recommendations()
        
        # Current recommendation
        self.current_recommendation = "Welcome to Emotion Detection System"
        self.last_update_time = time.time()
        
        # Demo mode flag
        self.demo_mode = False
    
    def load_face_cascade(self):
        """Attempt to load the face cascade classifier from multiple locations"""
        cascade_paths = [
            # OpenCV standard location
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else '',
            # Standard Linux paths
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            # Local paths
            'haarcascade_frontalface_default.xml',
            'data/haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            if path and os.path.exists(path):
                print(f"Loading cascade from: {path}")
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    return cascade
        
        print("WARNING: Could not find face cascade file.")
        print("Will use dummy face detection instead.")
        return None
    
    def init_recommendations(self):
        """Set up simple recommendations for each emotion"""
        self.recommendations = {
            "angry": [
                "Take 5 deep breaths",
                "Count to 10 slowly",
                "Step away briefly"
            ],
            "disgust": [
                "Focus on something pleasant",
                "Change your environment",
                "Think of something neutral"
            ],
            "fear": [
                "Remember you are safe now",
                "Focus on your breathing",
                "Ground yourself in the present"
            ],
            "happy": [
                "Savor this feeling",
                "Share your happiness",
                "Express gratitude"
            ],
            "sad": [
                "Be kind to yourself",
                "Reach out to someone",
                "Do something comforting"
            ],
            "surprise": [
                "Take a moment to process",
                "Consider your response",
                "Use this energy positively"
            ],
            "neutral": [
                "Check in with yourself",
                "Set an intention",
                "Good time for focused work"
            ]
        }
    
    def detect_dummy_face(self, frame):
        """Simple dummy face detection for testing when cascade fails"""
        height, width = frame.shape[:2]
        # Simply return a face in the center of the frame
        center_x = width // 2
        center_y = height // 2
        face_w = width // 4
        face_h = height // 4
        return [(center_x - face_w//2, center_y - face_h//2, face_w, face_h)]
    
    def initialize_camera(self):
        """Initialize the camera and return success status"""
        # Try multiple methods to open the camera
        self.cap = self.try_open_camera()
        
        # If camera opened successfully
        if self.cap is not None and self.cap.isOpened():
            # Try to set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.demo_mode = False
            return True
        else:
            print("Running in demo mode with static image...")
            self.demo_mode = True
            return True  # Return True even in demo mode so the UI can show demo
    
    def try_open_camera(self, index=0, max_attempts=3):
        """Try multiple methods to open the camera"""
        # First try: Default OpenCV method
        print(f"Trying to open camera {index} with default method...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print("Camera opened successfully with default method.")
            return cap
        
        # Second try: Explicitly disable GStreamer
        print("Trying to open camera with DirectShow backend...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DirectShow (Windows)
        if cap.isOpened():
            print("Camera opened successfully with DirectShow.")
            return cap
        
        # Third try: Try V4L2 backend (Linux)
        print("Trying to open camera with V4L2 backend...")
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            print("Camera opened successfully with V4L2.")
            return cap
        
        # Fourth try: Specify GStreamer pipeline directly
        print("Trying to open camera with explicit GStreamer pipeline...")
        gst_pipeline = f"v4l2src device=/dev/video{index} ! video/x-raw, width={self.width}, height={self.height} ! videoconvert ! appsink"
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("Camera opened successfully with GStreamer pipeline.")
            return cap
        
        # Try different device indices as last resort
        for i in range(max_attempts):
            if i == index:
                continue
            print(f"Trying camera index {i}...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera opened successfully with index {i}.")
                return cap
        
        print("Could not open camera with any method.")
        return None
    
    def process_frame(self):
        """Process a single frame and return it with emotion data"""
        if self.demo_mode:
            return self.process_demo_frame()
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            # Create a blank frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return True, frame, "error", "Camera error: Failed to grab frame"
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
            else:
                faces = self.detect_dummy_face(frame)
            
            # Process each face
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Simulate emotion detection
                    current_time = time.time()
                    if current_time - self.last_update_time > 5:
                        self.current_emotion = random.choice(self.emotions)
                        self.current_recommendation = random.choice(self.recommendations[self.current_emotion])
                        self.last_update_time = current_time
                    
                    # Add emotion text above face
                    cv2.putText(frame, f"Emotion: {self.current_emotion}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No faces detected
                cv2.putText(frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add recommendation at bottom of screen
            self.add_recommendation_to_frame(frame)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            # Add error message to frame
            cv2.putText(frame, f"Error: {str(e)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return True, frame, self.current_emotion, self.current_recommendation
    
    def process_demo_frame(self):
        """Process a demo frame when camera is not available"""
        # Create a black frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Simulate face detection
        faces = self.detect_dummy_face(frame)
        
        # Update emotion occasionally
        current_time = time.time()
        if current_time - self.last_update_time > 3:
            self.current_emotion = random.choice(self.emotions)
            self.current_recommendation = random.choice(self.recommendations[self.current_emotion])
            self.last_update_time = current_time
        
        # Draw face rectangle and emotion
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {self.current_emotion}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add text indicating demo mode
        cv2.putText(frame, "DEMO MODE - NO CAMERA", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add recommendation
        self.add_recommendation_to_frame(frame)
        
        return True, frame, self.current_emotion, self.current_recommendation
    
    def add_recommendation_to_frame(self, frame):
        """Add current recommendation to the bottom of the frame"""
        # Draw black rectangle at bottom
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
        
        # Add emotion and recommendation text
        cv2.putText(frame, f"Current emotion: {self.current_emotion.upper()}", 
                   (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Recommendation: {self.current_recommendation}", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def release_camera(self):
        """Release camera resources"""
        if self.cap is not None and not self.demo_mode:
            self.cap.release()


class EmotionDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize detector
        self.detector = SimpleEmotionDetection()
        
        # Set up UI
        self.initUI()
        
        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Detection running flag
        self.is_running = False
        
        # Emotion colors
        self.emotion_colors = {
            "angry": "#e74c3c",     # Red
            "disgust": "#9b59b6",   # Purple
            "fear": "#95a5a6",      # Gray
            "happy": "#2ecc71",     # Green
            "sad": "#3498db",       # Blue
            "surprise": "#f39c12",  # Orange
            "neutral": "#34495e"    # Dark Blue
        }
    
    def initUI(self):
        # Main window properties
        self.setWindowTitle('Emotion Detection System')
        self.setGeometry(100, 100, 800, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Header/title
        header = QLabel('Emotion Detection System')
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("background-color: #3498db; color: white; padding: 10px;")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(header)
        
        # Camera view
        self.camera_view = QLabel()
        self.camera_view.setMinimumSize(640, 480)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet("background-color: black;")
        self.camera_view.setText("Camera Feed Will Appear Here")
        main_layout.addWidget(self.camera_view)
        
        # Info section
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        info_frame.setStyleSheet("background-color: #f5f5f5; padding: 10px;")
        info_layout = QVBoxLayout(info_frame)
        
        # Emotion display
        emotion_layout = QHBoxLayout()
        emotion_label = QLabel("Detected Emotion:")
        emotion_label.setFont(QFont("Arial", 12))
        self.emotion_value = QLabel("Not detected")
        self.emotion_value.setFont(QFont("Arial", 12, QFont.Bold))
        emotion_layout.addWidget(emotion_label)
        emotion_layout.addWidget(self.emotion_value)
        emotion_layout.addStretch()
        info_layout.addLayout(emotion_layout)
        
        # Recommendation display
        recommendation_layout = QHBoxLayout()
        recommendation_label = QLabel("Recommendation:")
        recommendation_label.setFont(QFont("Arial", 12))
        self.recommendation_value = QLabel("Start detection to see recommendations")
        self.recommendation_value.setFont(QFont("Arial", 12))
        self.recommendation_value.setWordWrap(True)
        recommendation_layout.addWidget(recommendation_label)
        recommendation_layout.addWidget(self.recommendation_value, 1)
        info_layout.addLayout(recommendation_layout)
        
        main_layout.addWidget(info_frame)
        
        # Button controls
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.start_button.setStyleSheet("padding: 8px 16px;")
        
        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("padding: 8px 16px;")
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        self.quit_button.setStyleSheet("padding: 8px 16px;")
        
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.quit_button)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready to start")
    
    def start_detection(self):
        self.statusBar.showMessage("Starting camera...")
        
        # Initialize camera
        if self.detector.initialize_camera():
            self.statusBar.showMessage("Camera initialized successfully")
            
            # Update button states
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            # Start detection loop
            self.is_running = True
            self.timer.start(33)  # ~30 fps
        else:
            QMessageBox.critical(self, "Error", "Failed to initialize camera. Check camera connection.")
            self.statusBar.showMessage("Error: Camera initialization failed")
    
    def update_frame(self):
        if self.is_running:
            # Get frame from camera
            ret, frame, emotion, recommendation = self.detector.process_frame()
            
            if ret:
                # Update emotion and recommendation in UI
                self.emotion_value.setText(emotion.upper())
                
                # Set color based on emotion
                color = self.emotion_colors.get(emotion.lower(), "#2c3e50")
                self.emotion_value.setStyleSheet(f"color: {color};")
                
                self.recommendation_value.setText(recommendation)
                
                # Convert frame to QPixmap for display
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_img)
                
                # Update the image
                self.camera_view.setPixmap(pixmap)
                
                # Update status
                self.statusBar.showMessage(f"Running: {emotion} detected")
            else:
                self.statusBar.showMessage("Error: Failed to get frame")
    
    def stop_detection(self):
        if self.is_running:
            self.statusBar.showMessage("Stopping detection...")
            self.is_running = False
            self.timer.stop()
            
            # Release camera
            self.detector.release_camera()
            
            # Update button states
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.statusBar.showMessage("Detection stopped")
            
            # Clear camera view
            self.camera_view.clear()
            self.camera_view.setText("Camera Feed Will Appear Here")
    
    def closeEvent(self, event):
        # Override close event to handle cleanup
        if self.is_running:
            reply = QMessageBox.question(self, 'Quit', 
                                        'Are you sure you want to quit?',
                                        QMessageBox.Yes | QMessageBox.No, 
                                        QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.stop_detection()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionDetectionApp()
    window.show()
    sys.exit(app.exec_())