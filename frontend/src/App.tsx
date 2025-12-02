import { useState } from 'react'
import PipelineCanvas from './components/PipelineCanvas'
import ModulePanel from './components/ModulePanel'
import ResultViewer from './components/ResultViewer'
import './App.css'

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [resultImage, setResultImage] = useState<string | null>(null)
  const [showResultPanel, setShowResultPanel] = useState(false)

  const handleOpenResultPanel = (image: string | null) => {
    setResultImage(image)
    setShowResultPanel(true)
  }

  const handleCloseResultPanel = () => {
    setShowResultPanel(false)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>VisionBox</h1>
        <p>Computer Vision Toolkit</p>
      </header>
      <div className="app-content">
        <div className="sidebar">
          <ModulePanel />
        </div>
        <div className="main-area">
          <PipelineCanvas 
            selectedImage={selectedImage}
            onResultChange={setResultImage}
            onOpenResultPanel={handleOpenResultPanel}
          />
        </div>
        {showResultPanel && (
          <div className="result-panel">
            <ResultViewer 
              resultImage={resultImage} 
              onClose={handleCloseResultPanel}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default App

