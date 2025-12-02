"""
CV operation modules
"""
from modules.blur import BlurModule, GaussianBlurModule
from modules.edge import EdgeDetectionModule
from modules.threshold import ThresholdModule
from modules.transform import (
    GrayscaleModule, ResizeModule, RotateModule, FlipModule
)
from modules.adjustment import BrightnessModule, ContrastModule
from modules.face_detection import FaceDetectionModule, FaceDetectionEyesModule
from modules.feature_detection import (
    SIFTFeatureDetectionModule, ORBFeatureDetectionModule, FASTFeatureDetectionModule
)
from modules.morphology import MorphologyModule, MorphologyGradientModule
from modules.contour import ContourDetectionModule, ContourAnalysisModule
from modules.mediapipe_models import (
    MediaPipeFaceMeshModule, MediaPipeHandsModule, MediaPipePoseModule,
    MediaPipeHolisticModule, MediaPipeSelfieSegmentationModule
)
from modules.background_removal import AutoBackgroundRemoverModule, BackgroundRemoverU2NetModule
from modules.gan import GANImageGenerationModule
from modules.super_resolution import SuperResolutionModule
from modules.image_info import ImageInfoModule

__all__ = [
    "BlurModule",
    "GaussianBlurModule",
    "EdgeDetectionModule",
    "ThresholdModule",
    "GrayscaleModule",
    "ResizeModule",
    "RotateModule",
    "FlipModule",
    "BrightnessModule",
    "ContrastModule",
    "FaceDetectionModule",
    "FaceDetectionEyesModule",
    "SIFTFeatureDetectionModule",
    "ORBFeatureDetectionModule",
    "FASTFeatureDetectionModule",
    "MorphologyModule",
    "MorphologyGradientModule",
    "ContourDetectionModule",
    "ContourAnalysisModule",
    "MediaPipeFaceMeshModule",
    "MediaPipeHandsModule",
    "MediaPipePoseModule",
    "MediaPipeHolisticModule",
    "MediaPipeSelfieSegmentationModule",
    "AutoBackgroundRemoverModule",
    "BackgroundRemoverU2NetModule",
    "GANImageGenerationModule",
    "SuperResolutionModule",
    "ImageInfoModule",
]
