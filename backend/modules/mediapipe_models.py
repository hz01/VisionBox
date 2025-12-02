"""
MediaPipe model modules
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule


def _check_mediapipe_available():
    """Check if MediaPipe is available at runtime"""
    try:
        import mediapipe as mp
        return True
    except ImportError:
        return False


class MediaPipeFaceMeshModule(BaseCVModule):
    """Face Mesh detection with 468 facial landmarks"""
    
    @property
    def module_id(self) -> str:
        return "mediapipe_face_mesh"
    
    @property
    def display_name(self) -> str:
        return "MediaPipe Face Mesh"
    
    @property
    def description(self) -> str:
        return "Detect 468 facial landmarks using MediaPipe Face Mesh"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "max_num_faces",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 10,
                "description": "Maximum number of faces to detect"
            },
            {
                "name": "min_detection_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence for face detection"
            },
            {
                "name": "min_tracking_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence for face tracking"
            },
            {
                "name": "refine_landmarks",
                "type": "bool",
                "default": True,
                "description": "Refine landmarks for more accurate results"
            },
            {
                "name": "draw_landmarks",
                "type": "bool",
                "default": True,
                "description": "Draw facial landmarks on the image"
            },
            {
                "name": "draw_connections",
                "type": "bool",
                "default": True,
                "description": "Draw connections between landmarks"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if not _check_mediapipe_available():
            raise ValueError(
                "MediaPipe is not installed. Please install it with: pip install mediapipe"
            )
        
        import mediapipe as mp
        
        max_num_faces = kwargs.get("max_num_faces", 1)
        min_detection_confidence = kwargs.get("min_detection_confidence", 0.5)
        min_tracking_confidence = kwargs.get("min_tracking_confidence", 0.5)
        refine_landmarks = kwargs.get("refine_landmarks", True)
        draw_landmarks = kwargs.get("draw_landmarks", True)
        draw_connections = kwargs.get("draw_connections", True)
        
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        result = image.copy()
        
        if results.multi_face_landmarks and draw_landmarks:
            h, w = image.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                if draw_connections:
                    # Draw face mesh connections
                    mp.solutions.drawing_utils.draw_landmarks(
                        result,
                        face_landmarks,
                        mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        None,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
                else:
                    # Draw only landmarks
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(result, (x, y), 2, (0, 255, 0), -1)
        
        face_mesh.close()
        return result


class MediaPipeHandsModule(BaseCVModule):
    """Hand detection and landmarks using MediaPipe"""
    
    @property
    def module_id(self) -> str:
        return "mediapipe_hands"
    
    @property
    def display_name(self) -> str:
        return "MediaPipe Hands"
    
    @property
    def description(self) -> str:
        return "Detect hands and 21 hand landmarks using MediaPipe"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "max_num_hands",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 10,
                "description": "Maximum number of hands to detect"
            },
            {
                "name": "min_detection_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence for hand detection"
            },
            {
                "name": "min_tracking_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence for hand tracking"
            },
            {
                "name": "draw_landmarks",
                "type": "bool",
                "default": True,
                "description": "Draw hand landmarks on the image"
            },
            {
                "name": "draw_connections",
                "type": "bool",
                "default": True,
                "description": "Draw connections between landmarks"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if not _check_mediapipe_available():
            raise ValueError(
                "MediaPipe is not installed. Please install it with: pip install mediapipe"
            )
        
        import mediapipe as mp
        
        max_num_hands = kwargs.get("max_num_hands", 2)
        min_detection_confidence = kwargs.get("min_detection_confidence", 0.5)
        min_tracking_confidence = kwargs.get("min_tracking_confidence", 0.5)
        draw_landmarks = kwargs.get("draw_landmarks", True)
        draw_connections = kwargs.get("draw_connections", True)
        
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        result = image.copy()
        
        if results.multi_hand_landmarks and draw_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw_connections:
                    mp.solutions.drawing_utils.draw_landmarks(
                        result,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                else:
                    h, w = image.shape[:2]
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(result, (x, y), 3, (0, 255, 0), -1)
        
        hands.close()
        return result


class MediaPipePoseModule(BaseCVModule):
    """Body pose detection using MediaPipe"""
    
    @property
    def module_id(self) -> str:
        return "mediapipe_pose"
    
    @property
    def display_name(self) -> str:
        return "MediaPipe Pose"
    
    @property
    def description(self) -> str:
        return "Detect body pose with 33 landmarks using MediaPipe"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "min_detection_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence for pose detection"
            },
            {
                "name": "min_tracking_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence for pose tracking"
            },
            {
                "name": "model_complexity",
                "type": "select",
                "default": "1",
                "options": ["0", "1", "2"],
                "description": "Model complexity (0=lightest, 2=heaviest)"
            },
            {
                "name": "draw_landmarks",
                "type": "bool",
                "default": True,
                "description": "Draw pose landmarks on the image"
            },
            {
                "name": "draw_connections",
                "type": "bool",
                "default": True,
                "description": "Draw connections between landmarks"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if not _check_mediapipe_available():
            raise ValueError(
                "MediaPipe is not installed. Please install it with: pip install mediapipe"
            )
        
        import mediapipe as mp
        
        min_detection_confidence = kwargs.get("min_detection_confidence", 0.5)
        min_tracking_confidence = kwargs.get("min_tracking_confidence", 0.5)
        model_complexity = int(kwargs.get("model_complexity", "1"))
        draw_landmarks = kwargs.get("draw_landmarks", True)
        draw_connections = kwargs.get("draw_connections", True)
        
        pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        
        result = image.copy()
        
        if results.pose_landmarks and draw_landmarks:
            if draw_connections:
                mp.solutions.drawing_utils.draw_landmarks(
                    result,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            else:
                h, w = image.shape[:2]
                for landmark in results.pose_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(result, (x, y), 3, (0, 255, 0), -1)
        
        pose.close()
        return result


class MediaPipeHolisticModule(BaseCVModule):
    """Holistic detection: face, hands, and pose together"""
    
    @property
    def module_id(self) -> str:
        return "mediapipe_holistic"
    
    @property
    def display_name(self) -> str:
        return "MediaPipe Holistic"
    
    @property
    def description(self) -> str:
        return "Detect face, hands, and body pose simultaneously using MediaPipe"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "min_detection_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence for detection"
            },
            {
                "name": "min_tracking_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence for tracking"
            },
            {
                "name": "model_complexity",
                "type": "select",
                "default": "1",
                "options": ["0", "1", "2"],
                "description": "Model complexity (0=lightest, 2=heaviest)"
            },
            {
                "name": "draw_face",
                "type": "bool",
                "default": True,
                "description": "Draw face landmarks"
            },
            {
                "name": "draw_hands",
                "type": "bool",
                "default": True,
                "description": "Draw hand landmarks"
            },
            {
                "name": "draw_pose",
                "type": "bool",
                "default": True,
                "description": "Draw pose landmarks"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if not _check_mediapipe_available():
            raise ValueError(
                "MediaPipe is not installed. Please install it with: pip install mediapipe"
            )
        
        import mediapipe as mp
        
        min_detection_confidence = kwargs.get("min_detection_confidence", 0.5)
        min_tracking_confidence = kwargs.get("min_tracking_confidence", 0.5)
        model_complexity = int(kwargs.get("model_complexity", "1"))
        draw_face = kwargs.get("draw_face", True)
        draw_hands = kwargs.get("draw_hands", True)
        draw_pose = kwargs.get("draw_pose", True)
        
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_image)
        
        result = image.copy()
        
        # Draw face landmarks
        if results.face_landmarks and draw_face:
            mp.solutions.drawing_utils.draw_landmarks(
                result,
                results.face_landmarks,
                mp.solutions.holistic.FACEMESH_CONTOURS,
                None,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
        
        # Draw hand landmarks
        if draw_hands:
            if results.left_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    result,
                    results.left_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            if results.right_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    result,
                    results.right_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
        
        # Draw pose landmarks
        if results.pose_landmarks and draw_pose:
            mp.solutions.drawing_utils.draw_landmarks(
                result,
                results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        holistic.close()
        return result


class MediaPipeSelfieSegmentationModule(BaseCVModule):
    """Selfie segmentation using MediaPipe"""
    
    @property
    def module_id(self) -> str:
        return "mediapipe_selfie_segmentation"
    
    @property
    def display_name(self) -> str:
        return "MediaPipe Selfie Segmentation"
    
    @property
    def description(self) -> str:
        return "Segment person from background using MediaPipe"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "model_selection",
                "type": "select",
                "default": "0",
                "options": ["0", "1"],
                "description": "Model selection (0=general, 1=landscape)"
            },
            {
                "name": "min_detection_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence for detection"
            },
            {
                "name": "background_color",
                "type": "select",
                "default": "transparent",
                "options": ["transparent", "black", "white", "blur"],
                "description": "Background replacement option"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if not _check_mediapipe_available():
            raise ValueError(
                "MediaPipe is not installed. Please install it with: pip install mediapipe"
            )
        
        import mediapipe as mp
        
        model_selection = int(kwargs.get("model_selection", "0"))
        min_detection_confidence = kwargs.get("min_detection_confidence", 0.5)
        background_color = kwargs.get("background_color", "transparent")
        
        selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb_image)
        
        if results.segmentation_mask is None:
            return image.copy()
        
        # Convert mask to 3-channel
        mask = results.segmentation_mask
        mask_3channel = np.stack((mask,) * 3, axis=-1)
        
        # Apply background
        if background_color == "transparent":
            # Create RGBA image
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = (mask * 255).astype(np.uint8)
            # Convert back to BGR for consistency (alpha channel will be lost but that's okay)
            result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        elif background_color == "black":
            background = np.zeros_like(image)
            result = (mask_3channel * image + (1 - mask_3channel) * background).astype(np.uint8)
        elif background_color == "white":
            background = np.ones_like(image) * 255
            result = (mask_3channel * image + (1 - mask_3channel) * background).astype(np.uint8)
        else:  # blur
            background = cv2.GaussianBlur(image, (55, 55), 0)
            result = (mask_3channel * image + (1 - mask_3channel) * background).astype(np.uint8)
        
        selfie_segmentation.close()
        return result

