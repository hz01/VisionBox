"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class PipelineStep(BaseModel):
    """Single step in a pipeline"""
    id: str = Field(..., description="Unique identifier for this step")
    module: str = Field(..., description="Module ID to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Module parameters")


class PipelineRequest(BaseModel):
    """Request to execute a pipeline"""
    pipeline: List[PipelineStep] = Field(..., description="List of pipeline steps")
    stop_on_error: bool = Field(True, description="Stop execution on first error")


class PipelineResponse(BaseModel):
    """Response from pipeline execution"""
    success: bool
    image_base64: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)


class PipelineValidationResponse(BaseModel):
    """Response from pipeline validation"""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ModuleSchema(BaseModel):
    """Module schema information"""
    id: str
    display_name: str
    description: str
    category: str
    parameters: List[Dict[str, Any]]


class ModulesResponse(BaseModel):
    """Response containing all modules"""
    modules: List[ModuleSchema]
    categories: Dict[str, List[str]]

