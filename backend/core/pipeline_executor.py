"""
Pipeline execution engine
"""
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from core.module_registry import ModuleRegistry

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes pipelines of CV operations"""
    
    def __init__(self, module_registry: ModuleRegistry):
        self.module_registry = module_registry
    
    def execute(
        self, 
        image: np.ndarray, 
        pipeline: List[Dict[str, Any]],
        stop_on_error: bool = True
    ) -> Dict[str, Any]:
        """Execute a pipeline of operations
        
        Args:
            image: Input image as numpy array (BGR format)
            pipeline: List of operation definitions
                Format: [
                    {
                        "id": "node_id",
                        "module": "module_id",
                        "params": {...}
                    },
                    ...
                ]
            stop_on_error: If True, stop execution on first error
            
        Returns:
            Dict with keys:
            - success: bool
            - image: np.ndarray (processed image)
            - errors: List[str] (if any errors occurred)
            - execution_log: List[Dict] (execution details)
        """
        result = image.copy()
        errors = []
        execution_log = []
        
        for i, step in enumerate(pipeline):
            step_id = step.get("id", f"step_{i}")
            module_id = step.get("module")
            params = step.get("params", {})
            
            if not module_id:
                error_msg = f"Step {i} ({step_id}): Missing module ID"
                errors.append(error_msg)
                logger.error(error_msg)
                if stop_on_error:
                    break
                continue
            
            try:
                module = self.module_registry.get(module_id)
                validated_params = module.validate_parameters(**params)
                
                logger.info(f"Executing {module_id} with params: {validated_params}")
                result = module.process(result, **validated_params)
                
                execution_log.append({
                    "step_id": step_id,
                    "module_id": module_id,
                    "success": True,
                    "params": validated_params
                })
                
            except Exception as e:
                error_msg = f"Step {i} ({step_id}): Error in module '{module_id}': {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
                
                execution_log.append({
                    "step_id": step_id,
                    "module_id": module_id,
                    "success": False,
                    "error": str(e)
                })
                
                if stop_on_error:
                    break
        
        return {
            "success": len(errors) == 0,
            "image": result,
            "errors": errors,
            "execution_log": execution_log
        }
    
    def validate_pipeline(self, pipeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a pipeline configuration
        
        Args:
            pipeline: Pipeline definition
            
        Returns:
            Dict with keys:
            - valid: bool
            - errors: List[str]
            - warnings: List[str]
        """
        errors = []
        warnings = []
        
        if not pipeline:
            warnings.append("Pipeline is empty")
        
        for i, step in enumerate(pipeline):
            step_id = step.get("id", f"step_{i}")
            module_id = step.get("module")
            
            if not module_id:
                errors.append(f"Step {i} ({step_id}): Missing module ID")
                continue
            
            if not self.module_registry.exists(module_id):
                errors.append(f"Step {i} ({step_id}): Module '{module_id}' not found")
                continue
            
            try:
                module = self.module_registry.get(module_id)
                params = step.get("params", {})
                
                # Validate parameters
                try:
                    module.validate_parameters(**params)
                except ValueError as e:
                    errors.append(f"Step {i} ({step_id}): Invalid parameters - {str(e)}")
                
            except Exception as e:
                errors.append(f"Step {i} ({step_id}): {str(e)}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

