# VisionBox - Computer Vision Toolkit

An all-in-one computer vision toolkit with a visual pipeline builder UI. Upload images/videos and apply multiple CV operations in a visual, node-based interface.

## Features

- 🎨 **Visual Pipeline Builder**: Drag-and-drop interface using React Flow
- 🔧 **Modular Architecture**: Easy to add new CV operations
- 🖼️ **Image Processing**: Multiple OpenCV operations (blur, edge detection, threshold, transforms, etc.)
- 🚀 **FastAPI Backend**: High-performance Python backend
- 📦 **Extensible**: Simple module system for adding new operations

## Architecture

### Backend Structure
```
backend/
├── api/              # API routes
├── core/             # Core abstractions (BaseCVModule, ModuleRegistry, PipelineExecutor)
├── modules/          # CV operation modules
├── services/         # Business logic services
├── models/           # Pydantic schemas
├── utils/            # Utility functions
└── config/           # Configuration
```

### Frontend Structure
```
frontend/
├── src/
│   ├── components/   # React components
│   ├── services/     # API services
│   ├── store/        # Zustand state management
│   └── types/        # TypeScript types
```

## Getting Started

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Adding New CV Modules

The architecture is designed to be easily extensible. To add new CV modules, create a class that inherits from `BaseCVModule` in the `backend/modules/` directory and register it in `backend/main.py`. The module will automatically appear in the UI.

## Available Modules

### Filters
- **Blur**: Simple box blur
- **Gaussian Blur**: Gaussian blur filter
- **Threshold**: Binary threshold operations

### Transforms
- **Grayscale**: Convert to grayscale
- **Resize**: Resize image
- **Rotate**: Rotate image
- **Flip**: Flip image horizontally/vertically

### Detection
- **Edge Detection**: Canny edge detection

### Adjustments
- **Brightness**: Adjust brightness
- **Contrast**: Adjust contrast

## API Endpoints

- `GET /api/v1/modules` - Get all available modules
- `GET /api/v1/modules/{module_id}` - Get specific module details
- `POST /api/v1/pipeline/validate` - Validate pipeline configuration
- `POST /api/v1/pipeline/execute` - Execute pipeline on uploaded image

## License

See LICENSE file
