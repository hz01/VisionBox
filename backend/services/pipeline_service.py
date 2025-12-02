"""
Pipeline service for handling pipeline operations
"""
from typing import Dict, Any, List
import numpy as np
import logging
from core.pipeline_executor import PipelineExecutor
from core.module_registry import ModuleRegistry
from utils.image_utils import numpy_to_base64, validate_image

logger = logging.getLogger(__name__)


class PipelineService:
    """Service for pipeline operations"""
    
    def __init__(self, pipeline_executor: PipelineExecutor):
        self.pipeline_executor = pipeline_executor
    
    def execute_pipeline(
        self, 
        image: np.ndarray, 
        pipeline: List[Dict[str, Any]],
        stop_on_error: bool = True
    ) -> Dict[str, Any]:
        """Execute a pipeline on an image
        
        Args:
            image: Input image as numpy array
            pipeline: Pipeline definition
            stop_on_error: Whether to stop on first error
            
        Returns:
            Dict with execution results
        """
        # Validate input image
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            return {
                "success": False,
                "image_base64": None,
                "errors": [error_msg],
                "execution_log": []
            }
        
        # Execute pipeline
        result = self.pipeline_executor.execute(
            image, 
            pipeline, 
            stop_on_error=stop_on_error
        )
        
        # Convert result image to base64
        image_base64 = None
        if result["success"] and result["image"] is not None:
            try:
                image_base64 = numpy_to_base64(result["image"])
            except Exception as e:
                logger.error(f"Error converting image to base64: {e}")
                result["errors"].append(f"Error encoding result image: {str(e)}")
        
        return {
            "success": result["success"],
            "image_base64": image_base64,
            "errors": result["errors"],
            "execution_log": result["execution_log"]
        }
    
    def validate_pipeline(self, pipeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a pipeline configuration
        
        Args:
            pipeline: Pipeline definition
            
        Returns:
            Validation result
        """
        return self.pipeline_executor.validate_pipeline(pipeline)

