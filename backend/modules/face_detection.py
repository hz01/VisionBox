"""
Face detection modules with multiple model support
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule
import os


class FaceDetectionModule(BaseCVModule):
    """Face detection with multiple model support (Haar, MediaPipe, etc.)"""
    
    def __init__(self):
        # Initialize Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = None
            try:
                self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            except:
                pass
    
    @property
    def module_id(self) -> str:
        return "face_detection"
    
    @property
    def display_name(self) -> str:
        return "Face Detection"
    
    @property
    def description(self) -> str:
        return "Detect faces using multiple models: Haar Cascade, MediaPipe, etc."
    
    @property
    def category(self) -> str:
        return "detection"
    
    def _check_mediapipe_available(self):
        """Check if MediaPipe is available at runtime"""
        try:
            import mediapipe as mp
            return True
        except ImportError:
            return False
    
    @property
    def parameters(self):
        # Get available models - always include both, check at runtime
        available_models = ["haar", "mediapipe"]
        
        params = [
            {
                "name": "model",
                "type": "select",
                "default": "haar",
                "options": available_models,
                "description": "Face detection model to use"
            },
            # Haar Cascade parameters
            {
                "name": "haar_scale_factor",
                "type": "float",
                "default": 1.1,
                "min": 1.01,
                "max": 2.0,
                "description": "[Haar] Scale factor for image pyramid"
            },
            {
                "name": "haar_min_neighbors",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 20,
                "description": "[Haar] Minimum number of neighbors for detection"
            },
            {
                "name": "haar_min_size",
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 200,
                "description": "[Haar] Minimum face size in pixels"
            },
            # MediaPipe parameters
            {
                "name": "mediapipe_min_detection_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "[MediaPipe] Minimum confidence for detection"
            },
            {
                "name": "mediapipe_model_selection",
                "type": "select",
                "default": "0",
                "options": ["0", "1"],
                "description": "[MediaPipe] Model selection (0=short-range, 1=full-range)"
            },
            # Common parameters
            {
                "name": "detection_mode",
                "type": "select",
                "default": "multi",
                "options": ["single", "multi"],
                "description": "Detection mode: single face or multiple faces"
            },
            {
                "name": "max_faces",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 100,
                "description": "Maximum number of faces to detect (0 = unlimited, only used in multi mode)"
            },
            {
                "name": "draw_rectangles",
                "type": "bool",
                "default": True,
                "description": "Draw rectangles around detected faces"
            }
        ]
        
        return params
    
    def _detect_haar(self, image: np.ndarray, **kwargs) -> list:
        """Detect faces using Haar Cascade"""
        if self.face_cascade is None:
            raise ValueError("Haar Cascade classifier not available. Please ensure OpenCV is properly installed.")
        
        scale_factor = kwargs.get("haar_scale_factor", 1.1)
        min_neighbors = kwargs.get("haar_min_neighbors", 5)
        min_size = kwargs.get("haar_min_size", 30)
        detection_mode = kwargs.get("detection_mode", "multi")
        max_faces = kwargs.get("max_faces", 0)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size)
        )
        
        # Convert to list of (x, y, w, h) tuples
        faces_list = [(x, y, w, h) for (x, y, w, h) in faces]
        
        # Apply detection mode filter
        if detection_mode == "single":
            # Return only the first (largest) face
            if faces_list:
                # Sort by area (largest first) and take first
                faces_list.sort(key=lambda f: f[2] * f[3], reverse=True)
                faces_list = [faces_list[0]]
        elif max_faces > 0:
            # Limit to max_faces (largest ones)
            faces_list.sort(key=lambda f: f[2] * f[3], reverse=True)
            faces_list = faces_list[:max_faces]
        
        return faces_list
    
    def _detect_mediapipe(self, image: np.ndarray, **kwargs) -> list:
        """Detect faces using MediaPipe"""
        # Check if MediaPipe is available
        if not self._check_mediapipe_available():
            raise ValueError(
                "MediaPipe is not installed. Please install it with: pip install mediapipe\n"
                "Or use the 'haar' model instead."
            )
        
        # Import MediaPipe here to avoid import errors at module level
        import mediapipe as mp
        
        min_detection_confidence = kwargs.get("mediapipe_min_detection_confidence", 0.5)
        model_selection = int(kwargs.get("mediapipe_model_selection", "0"))
        detection_mode = kwargs.get("detection_mode", "multi")
        max_faces = kwargs.get("max_faces", 0)
        
        # Initialize MediaPipe face detection
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                # Store with area for sorting
                area = width * height
                faces.append((x, y, width, height, area))
        
        # Clean up
        face_detection.close()
        
        # Apply detection mode filter
        if detection_mode == "single":
            # Return only the first (largest) face
            if faces:
                # Sort by area (largest first) and take first
                faces.sort(key=lambda f: f[4], reverse=True)
                faces = [faces[0][:4]]  # Remove area from tuple
        elif max_faces > 0:
            # Limit to max_faces (largest ones)
            faces.sort(key=lambda f: f[4], reverse=True)
            faces = [f[:4] for f in faces[:max_faces]]  # Remove area from tuples
        else:
            # Remove area from tuples
            faces = [f[:4] for f in faces]
        
        return faces
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        model = kwargs.get("model", "haar")
        draw_rectangles = kwargs.get("draw_rectangles", True)
        
        # Detect faces based on selected model
        if model == "haar":
            faces = self._detect_haar(image, **kwargs)
        elif model == "mediapipe":
            faces = self._detect_mediapipe(image, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Draw rectangles on the original image
        result = image.copy()
        if draw_rectangles:
            for (x, y, w, h) in faces:
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return result


class FaceDetectionEyesModule(BaseCVModule):
    """Face and eye detection with multiple model support"""
    
    def __init__(self):
        # Initialize Haar Cascade
        cascade_path = cv2.data.haarcascades
        face_path = cascade_path + 'haarcascade_frontalface_default.xml'
        eye_path = cascade_path + 'haarcascade_eye.xml'
        
        self.face_cascade = None
        self.eye_cascade = None
        
        if os.path.exists(face_path):
            self.face_cascade = cv2.CascadeClassifier(face_path)
        if os.path.exists(eye_path):
            self.eye_cascade = cv2.CascadeClassifier(eye_path)
    
    def _check_mediapipe_available(self):
        """Check if MediaPipe is available at runtime"""
        try:
            import mediapipe as mp
            return True
        except ImportError:
            return False
    
    @property
    def module_id(self) -> str:
        return "face_eye_detection"
    
    @property
    def display_name(self) -> str:
        return "Face & Eye Detection"
    
    @property
    def description(self) -> str:
        return "Detect faces and eyes using Haar Cascade or MediaPipe"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        available_models = ["haar"]
        if self._check_mediapipe_available():
            available_models.append("mediapipe")
        
        return [
            {
                "name": "model",
                "type": "select",
                "default": "haar",
                "options": available_models,
                "description": "Detection model to use"
            },
            # Haar parameters
            {
                "name": "haar_scale_factor",
                "type": "float",
                "default": 1.1,
                "min": 1.01,
                "max": 2.0,
                "description": "[Haar] Scale factor for image pyramid"
            },
            {
                "name": "haar_min_neighbors",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 20,
                "description": "[Haar] Minimum number of neighbors for detection"
            },
            {
                "name": "haar_min_size",
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 200,
                "description": "[Haar] Minimum face size in pixels"
            },
            # MediaPipe parameters
            {
                "name": "mediapipe_min_detection_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "[MediaPipe] Minimum confidence for detection"
            },
            {
                "name": "mediapipe_min_tracking_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "[MediaPipe] Minimum confidence for tracking"
            },
            {
                "name": "mediapipe_model_selection",
                "type": "select",
                "default": "0",
                "options": ["0", "1"],
                "description": "[MediaPipe] Model selection (0=short-range, 1=full-range)"
            },
            # Common parameters
            {
                "name": "detection_mode",
                "type": "select",
                "default": "multi",
                "options": ["single", "multi"],
                "description": "Detection mode: single face or multiple faces"
            },
            {
                "name": "max_faces",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 100,
                "description": "Maximum number of faces to detect (0 = unlimited)"
            }
        ]
    
    def _detect_haar(self, image: np.ndarray, **kwargs) -> tuple:
        """Detect faces and eyes using Haar Cascade"""
        if self.face_cascade is None or self.eye_cascade is None:
            raise ValueError("Haar Cascade classifiers not available. Please ensure OpenCV is properly installed.")
        
        scale_factor = kwargs.get("haar_scale_factor", 1.1)
        min_neighbors = kwargs.get("haar_min_neighbors", 5)
        min_size = kwargs.get("haar_min_size", 30)
        detection_mode = kwargs.get("detection_mode", "multi")
        max_faces = kwargs.get("max_faces", 0)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size)
        )
        
        # Apply detection mode filter
        faces_list = [(x, y, w, h) for (x, y, w, h) in faces]
        if detection_mode == "single" and faces_list:
            faces_list.sort(key=lambda f: f[2] * f[3], reverse=True)
            faces_list = [faces_list[0]]
        elif max_faces > 0:
            faces_list.sort(key=lambda f: f[2] * f[3], reverse=True)
            faces_list = faces_list[:max_faces]
        
        # Detect eyes for each face
        faces_with_eyes = []
        for (x, y, w, h) in faces_list:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            eyes_list = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes]
            faces_with_eyes.append(((x, y, w, h), eyes_list))
        
        return faces_with_eyes
    
    def _detect_mediapipe(self, image: np.ndarray, **kwargs) -> tuple:
        """Detect faces and eyes using MediaPipe"""
        if not self._check_mediapipe_available():
            raise ValueError(
                "MediaPipe is not installed. Please install it with: pip install mediapipe\n"
                "Or use the 'haar' model instead."
            )
        
        import mediapipe as mp
        
        min_detection_confidence = kwargs.get("mediapipe_min_detection_confidence", 0.5)
        min_tracking_confidence = kwargs.get("mediapipe_min_tracking_confidence", 0.5)
        model_selection = int(kwargs.get("mediapipe_model_selection", "0"))
        detection_mode = kwargs.get("detection_mode", "multi")
        max_faces = kwargs.get("max_faces", 0)
        
        # Initialize MediaPipe face mesh for eye detection
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces if max_faces > 0 else 5,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Also use face detection for bounding boxes
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detection_results = face_detection.process(rgb_image)
        mesh_results = face_mesh.process(rgb_image)
        
        h, w = image.shape[:2]
        faces_with_eyes = []
        
        # Get face bounding boxes
        face_boxes = []
        if detection_results.detections:
            for detection in detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                area = width * height
                face_boxes.append((x, y, width, height, area))
        
        # Apply detection mode filter
        if detection_mode == "single" and face_boxes:
            face_boxes.sort(key=lambda f: f[4], reverse=True)
            face_boxes = [face_boxes[0]]
        elif max_faces > 0:
            face_boxes.sort(key=lambda f: f[4], reverse=True)
            face_boxes = face_boxes[:max_faces]
        
        # Get eye landmarks from face mesh
        if mesh_results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                if idx < len(face_boxes):
                    x, y, width, height, _ = face_boxes[idx]
                    
                    # Eye landmarks indices (left eye: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
                    # Right eye: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
                    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    
                    # Get eye bounding boxes
                    eyes = []
                    for eye_indices in [left_eye_indices, right_eye_indices]:
                        eye_points = [(int(face_landmarks.landmark[i].x * w), 
                                      int(face_landmarks.landmark[i].y * h)) 
                                     for i in eye_indices]
                        if eye_points:
                            min_x = min(p[0] for p in eye_points)
                            max_x = max(p[0] for p in eye_points)
                            min_y = min(p[1] for p in eye_points)
                            max_y = max(p[1] for p in eye_points)
                            eyes.append((min_x - x, min_y - y, max_x - min_x, max_y - min_y))
                    
                    faces_with_eyes.append(((x, y, width, height), eyes))
        
        face_detection.close()
        face_mesh.close()
        
        return faces_with_eyes
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        model = kwargs.get("model", "haar")
        
        # Detect faces and eyes based on selected model
        if model == "haar":
            faces_with_eyes = self._detect_haar(image, **kwargs)
        elif model == "mediapipe":
            faces_with_eyes = self._detect_mediapipe(image, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        result = image.copy()
        
        # Draw rectangles around faces and eyes
        for (x, y, w, h), eyes in faces_with_eyes:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(result, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
        
        return result

