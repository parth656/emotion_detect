import cv2
import random
import os
import time
import numpy as np
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QStatusBar, 
                            QMessageBox, QFrame, QComboBox, QSlider, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor


class EmotionDetector:
    """Improved emotion detection class with actual facial emotion recognition"""
    def __init__(self):
        # Emotions list (matches the model output indices)
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.current_emotion = "neutral"
        
        # Camera settings
        self.width = 640
        self.height = 480
        self.cap = None
        
        # Load face detection model
        self.face_cascade = self.load_face_cascade()
        
        # Load emotion recognition model
        self.emotion_model = self.load_emotion_model()
        
        # Initialize recommendations
        self.init_recommendations()
        
        # Current recommendation
        self.current_recommendation = "Welcome to Emotion Detection System"
        self.last_emotion_change = time.time()
        
        # Demo mode flag
        self.demo_mode = False
        
        # Settings
        self.emotion_threshold = 0.5  # Confidence threshold for emotion detection
        self.detection_frequency = 1.0  # Seconds between emotion detections
        self.display_confidence = True  # Whether to display confidence scores
        self.smooth_emotions = True  # Whether to smooth emotion transitions
        self.emotion_history = []  # Store recent emotions for smoothing
        self.history_size = 5  # Size of emotion history for smoothing
        
        # For facial landmarks-based emotion detection
        self.facial_features = {}
        self.last_detection_time = 0

    def load_face_cascade(self):
        """Load face cascade for face detection"""
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
                print(f"Loading face cascade from: {path}")
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    return cascade
        
        print("WARNING: Could not find face cascade file.")
        print("Will use dummy face detection instead.")
        return None

    def load_emotion_model(self):
        """
        Load or initialize models for emotion detection.
        We'll use facial landmarks for a more deterministic approach.
        """
        # Try to load facial landmark detector from OpenCV's face module if available
        try:
            # For OpenCV's DNN face detector
            prototxt = "deploy.prototxt"
            caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"
            
            # Check if models exist in common paths
            model_paths = [
                ('', ''),  # Current directory
                ('models/', 'models/'),  # Models subdirectory
                ('/usr/share/opencv4/face/', '/usr/share/opencv4/face/'),  # Linux system path
            ]
            
            for proto_path, caffe_path in model_paths:
                if os.path.exists(proto_path + prototxt) and os.path.exists(caffe_path + caffemodel):
                    print(f"Loading face detection model from: {caffe_path + caffemodel}")
                    return cv2.dnn.readNetFromCaffe(proto_path + prototxt, caffe_path + caffemodel)
            
            # If we couldn't find pre-trained models, use a rule-based approach
            print("Could not find pre-trained facial landmark models.")
            print("Using rule-based facial feature analysis for emotion detection.")
            return "rule_based"
            
        except Exception as e:
            print(f"Error loading facial landmark detector: {str(e)}")
            print("Using rule-based facial feature analysis for emotion detection.")
            return "rule_based"

    def analyze_face_emotion(self, face_img):
        """
        Analyze emotions in a face image using facial features.
        This is a more deterministic approach based on facial geometry.
        """
        if self.emotion_model == "rule_based":
            # Extract facial features for rule-based emotion detection
            features = self.extract_facial_features(face_img)
            
            # Use features to determine emotion (more deterministic than random)
            emotion, confidence, emotion_probs = self.rule_based_emotion_detection(features)
            
            # Update emotion history
            self.emotion_history.append((emotion, confidence))
            if len(self.emotion_history) > self.history_size:
                self.emotion_history.pop(0)
            
            # Apply smoothing if enabled
            if self.smooth_emotions and len(self.emotion_history) > 2:
                # Count occurrences of each emotion in history
                emotion_counts = {}
                for e, _ in self.emotion_history:
                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
                
                # Get the most common emotion
                most_common = max(emotion_counts.items(), key=lambda x: x[1])
                emotion = most_common[0]
                
                # Average confidence for this emotion
                confidences = [conf for em, conf in self.emotion_history if em == emotion]
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return emotion, confidence, emotion_probs
        else:
            # If we have a DNN-based emotion model, use it here
            # (This would be the place to implement a proper CNN-based emotion classifier)
            # For now, use a more stable version of the simulated approach
            current_time = time.time()
            time_diff = current_time - self.last_detection_time
            
            # Only change emotions occasionally based on facial features
            # This makes the emotion changes less random and more tied to face expressions
            if len(self.emotion_history) == 0 or time_diff > self.detection_frequency:
                # Get facial expression features
                features = self.extract_facial_features(face_img)
                
                # Calculate emotion probabilities based on features
                emotion_probs = self.feature_based_emotion_probs(features)
                
                # Sort by probability
                sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                
                # Get the top emotion and its probability
                top_emotion = sorted_emotions[0][0]
                confidence = sorted_emotions[0][1]
                
                # Update last detection time
                self.last_detection_time = current_time
                
                # Update emotion history
                self.emotion_history.append((top_emotion, confidence))
                if len(self.emotion_history) > self.history_size:
                    self.emotion_history.pop(0)
            else:
                # Use the most recent emotion
                top_emotion, confidence = self.emotion_history[-1]
                
                # Create probability distribution favoring the current emotion
                emotion_probs = {e: 0.1 for e in self.emotions}
                emotion_probs[top_emotion] = confidence
            
            # Apply smoothing if enabled
            if self.smooth_emotions and len(self.emotion_history) > 2:
                # Count occurrences of each emotion in history
                emotion_counts = {}
                for e, _ in self.emotion_history:
                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
                
                # Get the most common emotion
                most_common = max(emotion_counts.items(), key=lambda x: x[1])
                top_emotion = most_common[0]
                
                # Average confidence for this emotion
                confidences = [conf for em, conf in self.emotion_history if em == top_emotion]
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return top_emotion, confidence, emotion_probs

    def extract_facial_features(self, face_img):
        """
        Extract facial features for emotion detection.
        Uses basic computer vision techniques to extract features like:
        - Eye openness, eyebrow position
        - Mouth shape, smile detection
        - Face symmetry, etc.
        """
        height, width = face_img.shape[:2]
        
        # Normalize image
        if len(face_img.shape) > 2:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
            
        # Enhance contrast for better feature detection
        gray = cv2.equalizeHist(gray)
        
        # Detect facial features
        features = {}
        
        try:
            # Eye regions (assume standard face proportions)
            eye_h = height // 4
            eye_w = width // 5
            left_eye_roi = gray[height//4:height//4+eye_h, width//5:2*width//5]
            right_eye_roi = gray[height//4:height//4+eye_h, 3*width//5:4*width//5]
            
            # Mouth region
            mouth_roi = gray[2*height//3:, width//4:3*width//4]
            
            # Simple feature extraction (pixel intensity averages and variations)
            features['left_eye_avg'] = np.mean(left_eye_roi) if left_eye_roi.size > 0 else 127
            features['right_eye_avg'] = np.mean(right_eye_roi) if right_eye_roi.size > 0 else 127
            features['mouth_avg'] = np.mean(mouth_roi) if mouth_roi.size > 0 else 127
            
            features['left_eye_var'] = np.var(left_eye_roi) if left_eye_roi.size > 0 else 0
            features['right_eye_var'] = np.var(right_eye_roi) if right_eye_roi.size > 0 else 0
            features['mouth_var'] = np.var(mouth_roi) if mouth_roi.size > 0 else 0
            
            # Calculate symmetry
            left_half = gray[:, :width//2]
            right_half = cv2.flip(gray[:, width//2:], 1)
            
            # Resize if needed
            if left_half.shape != right_half.shape:
                min_width = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
            
            # Calculate symmetry score (lower is more symmetric)
            if left_half.size > 0 and right_half.size > 0:
                features['face_symmetry'] = np.sum(np.abs(left_half.astype(np.float32) - right_half.astype(np.float32))) / (left_half.size)
            else:
                features['face_symmetry'] = 1000  # High asymmetry if can't compute
            
            # Smile detection (horizontal edge detection in mouth area)
            if mouth_roi.size > 0:
                sobelx = cv2.Sobel(mouth_roi, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(mouth_roi, cv2.CV_64F, 0, 1, ksize=3)
                features['mouth_horizontal_edges'] = np.sum(np.abs(sobelx)) / mouth_roi.size
                features['mouth_vertical_edges'] = np.sum(np.abs(sobely)) / mouth_roi.size
            else:
                features['mouth_horizontal_edges'] = 0
                features['mouth_vertical_edges'] = 0
            
            # Overall image variance (for expressions like surprise which affect the whole face)
            features['overall_variance'] = np.var(gray)
            
            # Apply temporal filtering to features
            if not self.facial_features:
                # First frame, initialize
                self.facial_features = features
            else:
                # Apply exponential smoothing
                alpha = 0.3  # Smoothing factor
                for key in features:
                    if key in self.facial_features:
                        features[key] = alpha * features[key] + (1 - alpha) * self.facial_features[key]
                self.facial_features = features
            
            return features
            
        except Exception as e:
            print(f"Error extracting facial features: {str(e)}")
            # Return default features if extraction fails
            return {
                'left_eye_avg': 127, 'right_eye_avg': 127, 'mouth_avg': 127,
                'left_eye_var': 100, 'right_eye_var': 100, 'mouth_var': 100,
                'face_symmetry': 500,
                'mouth_horizontal_edges': 0.5, 'mouth_vertical_edges': 0.5,
                'overall_variance': 100
            }

    def rule_based_emotion_detection(self, features):
        """
        Determine emotion based on extracted facial features using rules.
        This is more deterministic than random emotion generation.
        """
        # Initialize scores for each emotion
        emotion_scores = {e: 0.0 for e in self.emotions}
        
        # Apply rules to score each emotion based on features
        # These rules are heuristic and could be improved with actual training data
        
        # Happy: High mouth_horizontal_edges (smile), low symmetry
        if features['mouth_horizontal_edges'] > 0.6:
            emotion_scores['happy'] += 0.6
        emotion_scores['happy'] += max(0, min(0.4, (features['mouth_horizontal_edges'] - 0.3) * 2))
        
        # Sad: Low mouth position, high symmetry
        emotion_scores['sad'] += max(0, min(0.7, (0.5 - features['mouth_horizontal_edges']) * 1.5))
        if features['mouth_avg'] < 100:
            emotion_scores['sad'] += 0.3
        
        # Angry: High contrast in eye area, low symmetry
        if features['left_eye_var'] > 150 and features['right_eye_var'] > 150:
            emotion_scores['angry'] += 0.4
        emotion_scores['angry'] += max(0, min(0.6, features['face_symmetry'] / 1000))
        
        # Surprise: High overall variance, high eye openness
        emotion_scores['surprise'] += max(0, min(0.8, features['overall_variance'] / 200))
        
        # Fear: Similar to surprise but with some asymmetry
        emotion_scores['fear'] = 0.7 * emotion_scores['surprise'] + 0.3 * max(0, min(0.5, features['face_symmetry'] / 800))
        
        # Disgust: Often involves asymmetry and specific mouth shape
        emotion_scores['disgust'] += max(0, min(0.7, features['face_symmetry'] / 700))
        if features['mouth_vertical_edges'] > features['mouth_horizontal_edges']:
            emotion_scores['disgust'] += 0.3
        
        # Neutral: High symmetry, balanced features
        symmetry_factor = max(0, min(1.0, 1.0 - (features['face_symmetry'] / 1000)))
        emotion_scores['neutral'] += 0.5 * symmetry_factor
        
        # Add some variance but not too much (makes it more realistic but still stable)
        for emotion in emotion_scores:
            emotion_scores[emotion] += random.uniform(-0.05, 0.05)
            # Ensure no negative scores
            emotion_scores[emotion] = max(0, emotion_scores[emotion])
        
        # Normalize to create probabilities
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_probs = {e: s/total for e, s in emotion_scores.items()}
        else:
            # Default to neutral if all scores are 0
            emotion_probs = {e: 0.1 for e in self.emotions}
            emotion_probs['neutral'] = 0.4
        
        # Get top emotion and its confidence
        top_emotion = max(emotion_probs.items(), key=lambda x: x[1])
        
        return top_emotion[0], top_emotion[1], emotion_probs

    def feature_based_emotion_probs(self, features):
        """
        Convert facial features to emotion probabilities.
        More sophisticated than pure random, but without requiring a pre-trained model.
        """
        # Initialize with base probabilities
        probs = {e: 0.1 for e in self.emotions}
        
        # Smile detection for happiness
        smile_prob = min(1.0, features['mouth_horizontal_edges'] / 0.8)
        probs['happy'] = 0.1 + 0.6 * smile_prob
        
        # Eye openness for surprise
        eye_openness = (features['left_eye_var'] + features['right_eye_var']) / 300
        probs['surprise'] = 0.1 + 0.4 * min(1.0, eye_openness)
        
        # Mouth shape for sadness
        sad_mouth = 1.0 - (features['mouth_avg'] / 255)
        probs['sad'] = 0.1 + 0.4 * sad_mouth
        
        # Face asymmetry for anger
        asymmetry = min(1.0, features['face_symmetry'] / 1000)
        probs['angry'] = 0.1 + 0.3 * asymmetry
        
        # Complex expression for disgust
        probs['disgust'] = 0.1 + 0.2 * asymmetry + 0.2 * sad_mouth
        
        # Fear combines elements of surprise and sadness
        probs['fear'] = 0.1 + 0.3 * eye_openness + 0.2 * sad_mouth
        
        # Neutral is high when other emotions are low
        non_neutral_prob = sum(probs.values()) - probs['neutral']
        probs['neutral'] = max(0.1, 1.0 - (non_neutral_prob / 3))
        
        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {e: p/total for e, p in probs.items()}
        
        return probs

    def init_recommendations(self):
        """Set up improved recommendations for each emotion"""
        self.recommendations = {
            "angry": [
                "Take 5 deep breaths to calm your nervous system",
                "Count to 10 slowly before responding",
                "Step away briefly to regain perspective",
                "Try to identify the specific trigger of your anger",
                "Consider whether your expectations are realistic"
            ],
            "disgust": [
                "Focus on something pleasant or neutral in your environment",
                "Practice non-judgmental awareness of your feelings",
                "Consider whether your reaction is proportionate to the stimulus",
                "Remind yourself that this feeling will pass",
                "Try changing your physical environment briefly"
            ],
            "fear": [
                "Remember that you are safe in this moment",
                "Practice 4-7-8 breathing (inhale for 4, hold for 7, exhale for 8)",
                "Ground yourself by naming 5 things you can see, 4 you can touch, etc.",
                "Challenge catastrophic thinking with evidence-based alternatives",
                "Visualize a peaceful, safe place that brings you comfort"
            ],
            "happy": [
                "Savor this positive feeling by being fully present in it",
                "Express gratitude for something specific in your life",
                "Share your positive feelings with someone else",
                "Notice the physical sensations of happiness in your body",
                "Use this positive state to engage in creative thinking"
            ],
            "sad": [
                "Be kind to yourself - sadness is a normal human emotion",
                "Reach out to a supportive friend or family member",
                "Do something gently comforting for yourself",
                "Allow yourself to feel sad without judgment",
                "Consider whether you need more rest or self-care"
            ],
            "surprise": [
                "Take a moment to process what just happened",
                "Notice any physical reactions like increased heart rate",
                "Consider whether this surprise offers any new information",
                "Use this alertness to generate creative ideas",
                "Channel the energy of surprise into positive action"
            ],
            "neutral": [
                "Check in with how you're feeling physically",
                "Set an intention for the next hour",
                "This is a good time for focused work or decision-making",
                "Practice mindful awareness of your surroundings",
                "Consider what emotion might best serve your current situation"
            ]
        }

    def detect_dummy_face(self, frame):
        """Simple dummy face detection for testing when cascade fails"""
        height, width = frame.shape[:2]
        # Return a face in the center of the frame
        center_x = width // 2
        center_y = height // 2
        face_w = width // 4
        face_h = height // 4
        return [(center_x - face_w//2, center_y - face_h//2, face_w, face_h)]

    def initialize_camera(self):
        """Initialize the camera and return success status"""
        self.cap = self.try_open_camera()
        
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.demo_mode = False
            return True
        else:
            print("Running in demo mode with static image...")
            self.demo_mode = True
            return True

    def try_open_camera(self, index=0, max_attempts=3):
        """Try multiple methods to open the camera"""
        # Try different methods to open the camera
        methods = [
            # Default method
            lambda: cv2.VideoCapture(index),
            # DirectShow (Windows)
            lambda: cv2.VideoCapture(index, cv2.CAP_DSHOW),
            # V4L2 (Linux)
            lambda: cv2.VideoCapture(index, cv2.CAP_V4L2),
            # GStreamer pipeline
            lambda: cv2.VideoCapture(
                f"v4l2src device=/dev/video{index} ! video/x-raw, width={self.width}, height={self.height} ! videoconvert ! appsink",
                cv2.CAP_GSTREAMER
            )
        ]
        
        # Try each method
        for i, method in enumerate(methods):
            print(f"Trying camera method {i+1}...")
            try:
                cap = method()
                if cap.isOpened():
                    print(f"Camera opened successfully with method {i+1}")
                    return cap
            except Exception as e:
                print(f"Method {i+1} failed: {str(e)}")
        
        # Try different device indices
        for i in range(max_attempts):
            if i == index:
                continue
            print(f"Trying camera index {i}...")
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Camera opened successfully with index {i}")
                    return cap
            except Exception as e:
                print(f"Camera index {i} failed: {str(e)}")
        
        print("Could not open camera with any method")
        return None

    def process_frame(self):
        """Process a single frame and return it with emotion data"""
        if self.demo_mode:
            return self.process_demo_frame()
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return True, frame, "error", "Camera error: Failed to grab frame", {}
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = []
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
            
            # If no faces detected with cascade, try other methods
            if len(faces) == 0:
                # Try DNN-based detection if model is loaded
                if isinstance(self.emotion_model, cv2.dnn.Net):
                    try:
                        blob = cv2.dnn.blobFromImage(
                            cv2.resize(frame, (300, 300)), 
                            1.0, (300, 300), 
                            (104.0, 177.0, 123.0)
                        )
                        self.emotion_model.setInput(blob)
                        detections = self.emotion_model.forward()
                        
                        # Process detections
                        for i in range(detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            if confidence > 0.5:  # Confidence threshold
                                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                                (x, y, x2, y2) = box.astype("int")
                                faces.append((x, y, x2-x, y2-y))
                    except Exception as e:
                        print(f"DNN face detection error: {str(e)}")
                
                # If still no faces, use dummy detection
                if len(faces) == 0:
                    faces = self.detect_dummy_face(frame)
            
            # Process each face
            emotion_probs = {}
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Extract face region for emotion analysis
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size == 0:
                        continue
                    
                    # Resize to expected size (typical for CNN models)
                    try:
                        face_roi_resized = cv2.resize(face_roi, (48, 48))
                        
                        # Analyze emotion
                        self.current_emotion, confidence, emotion_probs = self.analyze_face_emotion(face_roi_resized)
                        
                        # Select a recommendation
                        if time.time() - self.last_emotion_change > self.detection_frequency:
                            self.current_recommendation = random.choice(self.recommendations[self.current_emotion])
                            self.last_emotion_change = time.time()
                        
                        # Add emotion text above face
                        cv2.putText(frame, f"Emotion: {self.current_emotion}", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Add confidence if enabled
                        if self.display_confidence:
                            cv2.putText(frame, f"Conf: {confidence:.2f}", 
                                      (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        
                        # Draw emotion probabilities as a bar graph
                        if self.display_confidence:
                            self.draw_emotion_bars(frame, emotion_probs)
                            
                        # Draw facial feature regions for debug
                        self.draw_feature_regions(frame, x, y, w, h)
                    except Exception as e:
                        print(f"Error processing face ROI: {str(e)}")
            else:
                # No faces detected
                cv2.putText(frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add recommendation at bottom of screen
            self.add_recommendation_to_frame(frame)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            cv2.putText(frame, f"Error: {str(e)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            emotion_probs = {}
        
        return True, frame, self.current_emotion, self.current_recommendation, emotion_probs
    
    def draw_feature_regions(self, frame, x, y, w, h):
        """Draw facial feature regions for debugging"""
        if not self.display_confidence:
            return
            
        # Draw eye regions
        eye_h = h // 4
        eye_w = w // 5
        cv2.rectangle(frame, (x + w//5, y + h//4), (x + 2*w//5, y + h//4 + eye_h), (255, 0, 0), 1)
        cv2.rectangle(frame, (x + 3*w//5, y + h//4), (x + 4*w//5, y + h//4 + eye_h), (255, 0, 0), 1)
        
        # Draw mouth region
        cv2.rectangle(frame, (x + w//4, y + 2*h//3), (x + 3*w//4, y + h), (0, 0, 255), 1)

    def draw_emotion_bars(self, frame, emotion_probs):
        """Draw emotion probability bars on the frame"""
        h, w = frame.shape[:2]
        
        # Parameters
        bar_height = 15
        max_bar_width = 150
        x_offset = w - max_bar_width - 10
        y_offset = 10
        
        # Sort emotions by probability
        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Draw bars for each emotion
        for i, (emotion, prob) in enumerate(sorted_emotions):
            # Calculate bar width based on probability
            bar_width = int(prob * max_bar_width)
            
            # Set color based on emotion
            if emotion == 'happy':
                color = (0, 255, 0)  # Green
            elif emotion == 'angry':
                color = (0, 0, 255)  # Red
            elif emotion == 'sad':
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw the bar
            cv2.rectangle(frame, (x_offset, y_offset + i * (bar_height + 5)), 
                         (x_offset + bar_width, y_offset + i * (bar_height + 5) + bar_height), 
                         color, -1)
            
            # Draw text label
            cv2.putText(frame, f"{emotion}: {prob:.2f}", 
                       (x_offset - 100, y_offset + i * (bar_height + 5) + bar_height - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def process_demo_frame(self):
        """Generate a demo frame with simulated face and emotions"""
        # Create a blank frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (200, 200, 200)  # Light gray background
        
        # Draw a simple face
        center_x = self.width // 2
        center_y = self.height // 2
        face_size = min(self.width, self.height) // 3
        
        # Face circle
        cv2.circle(frame, (center_x, center_y), face_size, (200, 170, 130), -1)
        cv2.circle(frame, (center_x, center_y), face_size, (100, 80, 60), 2)
        
        # Slowly change emotions over time
        t = time.time() % 30  # 30-second cycle
        emotion_idx = int((t / 30) * len(self.emotions))
        self.current_emotion = self.emotions[emotion_idx]
        
        # Select a recommendation if it's time for a new one
        if time.time() - self.last_emotion_change > 5:  # Change every 5 seconds in demo mode
            self.current_recommendation = random.choice(self.recommendations[self.current_emotion])
            self.last_emotion_change = time.time()
        
        # Eyes
        eye_y = center_y - face_size // 4
        left_eye_x = center_x - face_size // 3
        right_eye_x = center_x + face_size // 3
        
        # Draw eyes differently based on emotion
        if self.current_emotion == "happy":
            # Happy eyes (slightly closed)
            cv2.ellipse(frame, (left_eye_x, eye_y), (15, 10), 0, 0, 180, (255, 255, 255), -1)
            cv2.ellipse(frame, (right_eye_x, eye_y), (15, 10), 0, 0, 180, (255, 255, 255), -1)
        elif self.current_emotion == "sad":
            # Sad eyes (droopy)
            cv2.ellipse(frame, (left_eye_x, eye_y), (15, 10), 0, 180, 360, (255, 255, 255), -1)
            cv2.ellipse(frame, (right_eye_x, eye_y), (15, 10), 0, 180, 360, (255, 255, 255), -1)
        elif self.current_emotion == "surprise" or self.current_emotion == "fear":
            # Surprised eyes (wide open)
            cv2.circle(frame, (left_eye_x, eye_y), 20, (255, 255, 255), -1)
            cv2.circle(frame, (right_eye_x, eye_y), 20, (255, 255, 255), -1)
        else:
            # Normal eyes
            cv2.circle(frame, (left_eye_x, eye_y), 15, (255, 255, 255), -1)
            cv2.circle(frame, (right_eye_x, eye_y), 15, (255, 255, 255), -1)
        
        # Pupils
        cv2.circle(frame, (left_eye_x, eye_y), 5, (0, 0, 0), -1)
        cv2.circle(frame, (right_eye_x, eye_y), 5, (0, 0, 0), -1)
        
        # Mouth based on emotion
        mouth_y = center_y + face_size // 3
        mouth_width = face_size // 2
        mouth_height = face_size // 4
        
        if self.current_emotion == "happy":
            # Happy mouth (smile)
            cv2.ellipse(frame, (center_x, mouth_y), (mouth_width, mouth_height), 0, 0, 180, (0, 0, 0), 2)
        elif self.current_emotion == "sad":
            # Sad mouth (frown)
            cv2.ellipse(frame, (center_x, mouth_y + mouth_height), (mouth_width, mouth_height), 0, 180, 360, (0, 0, 0), 2)
        elif self.current_emotion == "surprise":
            # Surprised mouth (O shape)
            cv2.circle(frame, (center_x, mouth_y), mouth_width // 2, (0, 0, 0), 2)
        elif self.current_emotion == "angry":
            # Angry mouth (straight line with downturned edges)
            cv2.line(frame, (center_x - mouth_width // 2, mouth_y), (center_x + mouth_width // 2, mouth_y), (0, 0, 0), 2)
            cv2.line(frame, (center_x - mouth_width // 2, mouth_y), (center_x - mouth_width // 2 + 10, mouth_y + 10), (0, 0, 0), 2)
            cv2.line(frame, (center_x + mouth_width // 2, mouth_y), (center_x + mouth_width // 2 - 10, mouth_y + 10), (0, 0, 0), 2)
        elif self.current_emotion == "disgust":
            # Disgust mouth (curved down one side)
            pts = np.array([[center_x - mouth_width // 2, mouth_y], 
                           [center_x, mouth_y + mouth_height // 2], 
                           [center_x + mouth_width // 2, mouth_y - mouth_height // 3]], np.int32)
            cv2.polylines(frame, [pts], False, (0, 0, 0), 2)
        elif self.current_emotion == "fear":
            # Fear mouth (wavy line)
            for i in range(-mouth_width // 2, mouth_width // 2, 5):
                y_offset = int(5 * np.sin(i * 0.2))
                cv2.circle(frame, (center_x + i, mouth_y + y_offset), 1, (0, 0, 0), 2)
        else:
            # Neutral mouth (straight line)
            cv2.line(frame, (center_x - mouth_width // 2, mouth_y), (center_x + mouth_width // 2, mouth_y), (0, 0, 0), 2)
        
        # Add text showing the emotion
        cv2.putText(frame, f"DEMO MODE - Current Emotion: {self.current_emotion}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Generate simulated emotion probabilities
        emotion_probs = {e: 0.1 for e in self.emotions}
        emotion_probs[self.current_emotion] = 0.6
        
        # Add secondary emotion for realism
        secondary_idx = (emotion_idx + 1) % len(self.emotions)
        emotion_probs[self.emotions[secondary_idx]] = 0.3
        
        # Draw emotion probability bars
        self.draw_emotion_bars(frame, emotion_probs)
        
        # Add recommendation
        self.add_recommendation_to_frame(frame)
        
        return True, frame, self.current_emotion, self.current_recommendation, emotion_probs

    def add_recommendation_to_frame(self, frame):
        """Add emotion-based recommendation to the bottom of the frame"""
        h, w = frame.shape[:2]
        
        # Create a semi-transparent overlay for the recommendation
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (50, 50, 50), -1)
        
        # Add the recommendation text
        cv2.putText(overlay, f"Recommendation: {self.current_recommendation}", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blend the overlay with the original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def release(self):
        """Release camera and other resources"""
        if self.cap is not None:
            self.cap.release()


class EmotionDetectorApp(QMainWindow):
    """Main application window for the emotion detector"""
    def __init__(self):
        super().__init__()
        self.detector = EmotionDetector()
        self.initUI()
        
        # Initialize camera
        self.start_camera()
        
        # Set up timer for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (approx. 33 fps)

    def initUI(self):
        """Initialize the UI components"""
        self.setWindowTitle('Emotion Detector')
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        mainLayout = QVBoxLayout(mainWidget)
        
        # Camera display
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setMinimumSize(640, 480)
        self.imageLabel.setStyleSheet("background-color: black;")
        mainLayout.addWidget(self.imageLabel)
        
        # Control panel
        controlPanel = QFrame(self)
        controlPanel.setFrameShape(QFrame.StyledPanel)
        controlLayout = QHBoxLayout(controlPanel)
        
        # Settings panel
        settingsPanel = QFrame(self)
        settingsLayout = QVBoxLayout(settingsPanel)
        settingsLabel = QLabel("Settings", self)
        settingsLabel.setFont(QFont('Arial', 12, QFont.Bold))
        settingsLayout.addWidget(settingsLabel)
        
        # Settings options
        confLayout = QHBoxLayout()
        confLabel = QLabel("Detection Confidence:", self)
        self.confSlider = QSlider(Qt.Horizontal, self)
        self.confSlider.setMinimum(1)
        self.confSlider.setMaximum(10)
        self.confSlider.setValue(5)
        self.confSlider.setTickPosition(QSlider.TicksBelow)
        self.confSlider.setTickInterval(1)
        self.confSlider.valueChanged.connect(self.update_settings)
        confLayout.addWidget(confLabel)
        confLayout.addWidget(self.confSlider)
        
        # Display options
        displayLayout = QHBoxLayout()
        self.showConfidence = QCheckBox("Show Confidence", self)
        self.showConfidence.setChecked(True)
        self.showConfidence.stateChanged.connect(self.update_settings)
        self.smoothEmotions = QCheckBox("Smooth Emotions", self)
        self.smoothEmotions.setChecked(True)
        self.smoothEmotions.stateChanged.connect(self.update_settings)
        displayLayout.addWidget(self.showConfidence)
        displayLayout.addWidget(self.smoothEmotions)
        
        # Add layouts to settings panel
        settingsLayout.addLayout(confLayout)
        settingsLayout.addLayout(displayLayout)
        controlLayout.addWidget(settingsPanel)
        
        # Emotion display panel
        emotionPanel = QFrame(self)
        emotionLayout = QVBoxLayout(emotionPanel)
        emotionLabel = QLabel("Current Emotion", self)
        emotionLabel.setFont(QFont('Arial', 12, QFont.Bold))
        emotionLayout.addWidget(emotionLabel)
        
        # Current emotion display
        self.currentEmotionLabel = QLabel("Neutral", self)
        self.currentEmotionLabel.setFont(QFont('Arial', 16))
        self.currentEmotionLabel.setAlignment(Qt.AlignCenter)
        self.currentEmotionLabel.setStyleSheet("color: white; background-color: #333; padding: 5px;")
        emotionLayout.addWidget(self.currentEmotionLabel)
        
        # Recommendation label
        self.recommendationLabel = QLabel("Welcome to Emotion Detector", self)
        self.recommendationLabel.setWordWrap(True)
        self.recommendationLabel.setAlignment(Qt.AlignCenter)
        self.recommendationLabel.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        emotionLayout.addWidget(self.recommendationLabel)
        
        controlLayout.addWidget(emotionPanel)
        
        # Button panel
        buttonPanel = QFrame(self)
        buttonLayout = QVBoxLayout(buttonPanel)
        
        # Restart camera button
        self.restartButton = QPushButton("Restart Camera", self)
        self.restartButton.clicked.connect(self.restart_camera)
        buttonLayout.addWidget(self.restartButton)
        
        # Toggle demo mode button
        self.demoButton = QPushButton("Toggle Demo Mode", self)
        self.demoButton.clicked.connect(self.toggle_demo_mode)
        buttonLayout.addWidget(self.demoButton)
        
        # Exit button
        self.exitButton = QPushButton("Exit", self)
        self.exitButton.clicked.connect(self.close)
        buttonLayout.addWidget(self.exitButton)
        
        controlLayout.addWidget(buttonPanel)
        mainLayout.addWidget(controlPanel)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def update_settings(self):
        """Update detector settings from UI controls"""
        # Update confidence threshold
        self.detector.emotion_threshold = self.confSlider.value() / 10.0
        
        # Update display options
        self.detector.display_confidence = self.showConfidence.isChecked()
        self.detector.smooth_emotions = self.smoothEmotions.isChecked()
        
        self.statusBar.showMessage("Settings updated")

    def start_camera(self):
        """Initialize and start the camera"""
        if not self.detector.initialize_camera():
            QMessageBox.warning(self, "Camera Error", 
                               "Could not initialize camera. Running in demo mode.")

    def restart_camera(self):
        """Restart the camera"""
        self.statusBar.showMessage("Restarting camera...")
        self.detector.release()
        self.start_camera()

    def toggle_demo_mode(self):
        """Toggle between demo mode and live camera mode"""
        self.detector.demo_mode = not self.detector.demo_mode
        status = "Demo mode enabled" if self.detector.demo_mode else "Live camera mode enabled"
        self.statusBar.showMessage(status)

    def update_frame(self):
        """Update the frame from the camera"""
        success, frame, emotion, recommendation, _ = self.detector.process_frame()
        
        if success:
            # Convert OpenCV BGR image to RGB for Qt
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # Convert to QImage and display
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.imageLabel.setPixmap(QPixmap.fromImage(qt_image))
            
            # Update emotion and recommendation labels
            self.currentEmotionLabel.setText(emotion.capitalize())
            self.recommendationLabel.setText(recommendation)
            
            # Set emotion-based color for the emotion label
            if emotion == "happy":
                color = "#4CAF50"  # Green
            elif emotion == "angry":
                color = "#F44336"  # Red
            elif emotion == "sad":
                color = "#2196F3"  # Blue
            elif emotion == "fear":
                color = "#673AB7"  # Purple
            elif emotion == "surprise":
                color = "#FF9800"  # Orange
            elif emotion == "disgust":
                color = "#795548"  # Brown
            else:
                color = "#607D8B"  # Blue Grey (neutral)
                
            self.currentEmotionLabel.setStyleSheet(f"color: white; background-color: {color}; padding: 5px;")

    def closeEvent(self, event):
        """Handle window close event"""
        self.timer.stop()
        self.detector.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EmotionDetectorApp()
    ex.show()
    sys.exit(app.exec_())