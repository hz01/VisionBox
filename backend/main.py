"""
VisionBox API - Main application
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging

from api.routes import router
from core.module_registry import ModuleRegistry
from modules import (
    BlurModule, GaussianBlurModule, EdgeDetectionModule,
    ThresholdModule, GrayscaleModule, ResizeModule,
    RotateModule, FlipModule, BrightnessModule, ContrastModule
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VisionBox API",
    description="Computer Vision Toolkit API",
    version="1.0.0"
)

# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    errors = exc.errors()
    error_messages = []
    for error in errors:
        loc = " -> ".join(str(l) for l in error["loc"])
        msg = error["msg"]
        error_messages.append(f"{loc}: {msg}")
    
    logger.error(f"Validation error: {error_messages}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "; ".join(error_messages)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize module registry
module_registry = ModuleRegistry()

# Register all modules
modules_to_register = [
    BlurModule(),
    GaussianBlurModule(),
    EdgeDetectionModule(),
    ThresholdModule(),
    GrayscaleModule(),
    ResizeModule(),
    RotateModule(),
    FlipModule(),
    BrightnessModule(),
    ContrastModule(),
]

for module in modules_to_register:
    try:
        module_registry.register(module)
        logger.info(f"Registered module: {module.module_id}")
    except Exception as e:
        logger.error(f"Failed to register module {module.module_id}: {e}")

# Include API routes
app.include_router(router)

@app.get("/")
async def root():
    return {
        "message": "VisionBox API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

