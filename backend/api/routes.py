"""
API routes
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import numpy as np
import logging

from models.schemas import (
    PipelineRequest, PipelineResponse, PipelineValidationResponse,
    ModulesResponse
)
from services.pipeline_service import PipelineService
from services.module_service import ModuleService
from utils.image_utils import file_to_numpy, validate_image
from core.module_registry import ModuleRegistry
from core.pipeline_executor import PipelineExecutor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["api"])


# Dependency injection
def get_module_registry() -> ModuleRegistry:
    """Get module registry instance"""
    from main import module_registry
    return module_registry


def get_pipeline_executor(registry: ModuleRegistry = Depends(get_module_registry)) -> PipelineExecutor:
    """Get pipeline executor instance"""
    return PipelineExecutor(registry)


def get_pipeline_service(executor: PipelineExecutor = Depends(get_pipeline_executor)) -> PipelineService:
    """Get pipeline service instance"""
    return PipelineService(executor)


def get_module_service(registry: ModuleRegistry = Depends(get_module_registry)) -> ModuleService:
    """Get module service instance"""
    return ModuleService(registry)


@router.get("/modules", response_model=ModulesResponse)
async def get_modules(service: ModuleService = Depends(get_module_service)):
    """Get all available CV modules"""
    try:
        result = service.get_all_modules()
        return ModulesResponse(**result)
    except Exception as e:
        logger.error(f"Error getting modules: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modules/{module_id}")
async def get_module(module_id: str, service: ModuleService = Depends(get_module_service)):
    """Get a specific module by ID"""
    try:
        return service.get_module(module_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting module {module_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/validate", response_model=PipelineValidationResponse)
async def validate_pipeline(
    request: PipelineRequest,
    service: PipelineService = Depends(get_pipeline_service)
):
    """Validate a pipeline configuration"""
    try:
        pipeline = [step.dict() for step in request.pipeline]
        result = service.validate_pipeline(pipeline)
        return PipelineValidationResponse(**result)
    except Exception as e:
        logger.error(f"Error validating pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/execute", response_model=PipelineResponse)
async def execute_pipeline(
    file: UploadFile = File(...),
    pipeline: Optional[str] = Form(None),
    service: PipelineService = Depends(get_pipeline_service)
):
    """Execute a pipeline on an uploaded image
    
    Pipeline should be sent as a JSON string in the 'pipeline' form field
    """
    error_detail = None
    try:
        # Read and decode image
        file_bytes = await file.read()
        
        if not file_bytes:
            error_detail = "No file uploaded or file is empty"
            logger.error(error_detail)
            raise HTTPException(status_code=400, detail=error_detail)
        
        try:
            image = file_to_numpy(file_bytes)
        except ValueError as e:
            error_detail = f"Invalid image file: {str(e)}"
            logger.error(error_detail)
            raise HTTPException(status_code=400, detail=error_detail)
        except Exception as e:
            error_detail = f"Error processing image: {str(e)}"
            logger.error(error_detail, exc_info=True)
            raise HTTPException(status_code=400, detail=error_detail)
        
        # Validate image
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            error_detail = f"Invalid image: {error_msg}"
            logger.error(error_detail)
            raise HTTPException(status_code=400, detail=error_detail)
        
        # Parse pipeline
        import json
        if not pipeline:
            error_detail = "Pipeline is required"
            logger.error(error_detail)
            raise HTTPException(status_code=400, detail=error_detail)
        
        try:
            pipeline_data = json.loads(pipeline)
            if not isinstance(pipeline_data, list):
                error_detail = "Pipeline must be a list"
                logger.error(error_detail)
                raise HTTPException(status_code=400, detail=error_detail)
        except json.JSONDecodeError as e:
            error_detail = f"Invalid JSON in pipeline: {str(e)}"
            logger.error(error_detail)
            raise HTTPException(status_code=400, detail=error_detail)
        except ValueError as e:
            error_detail = str(e)
            logger.error(f"Pipeline validation error: {error_detail}")
            raise HTTPException(status_code=400, detail=error_detail)
        
        # Execute pipeline
        result = service.execute_pipeline(image, pipeline_data)
        return PipelineResponse(**result)
        
    except HTTPException as he:
        # Re-raise HTTPException with proper detail
        logger.error(f"HTTPException raised: {he.detail}")
        raise
    except Exception as e:
        error_detail = f"Internal server error: {str(e)}"
        logger.error(error_detail, exc_info=True)
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/pipeline/execute/json", response_model=PipelineResponse)
async def execute_pipeline_json(
    request: PipelineRequest,
    file: UploadFile = File(...),
    service: PipelineService = Depends(get_pipeline_service)
):
    """Execute a pipeline on an uploaded image (JSON body version)"""
    try:
        # Read and decode image
        file_bytes = await file.read()
        
        if not file_bytes:
            raise HTTPException(status_code=400, detail="No file uploaded or file is empty")
        
        try:
            image = file_to_numpy(file_bytes)
        except ValueError as e:
            logger.error(f"Error decoding image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
        # Validate image
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            logger.error(f"Image validation failed: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Invalid image: {error_msg}")
        
        # Convert request to pipeline format
        pipeline = [step.dict() for step in request.pipeline]
        
        # Execute pipeline
        result = service.execute_pipeline(
            image, 
            pipeline, 
            stop_on_error=request.stop_on_error
        )
        return PipelineResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

