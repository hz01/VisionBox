import { Download, X } from 'lucide-react'
import './ResultViewer.css'

interface ResultViewerProps {
  resultImage: string | null
  onClose?: () => void
}

function ResultViewer({ resultImage, onClose }: ResultViewerProps) {
  const handleDownload = () => {
    if (!resultImage) return
    
    const link = document.createElement('a')
    link.href = resultImage
    link.download = `visionbox-result-${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="result-viewer">
      <div className="result-header">
        <h2>Result</h2>
        <div className="header-actions">
          {resultImage && (
            <button onClick={handleDownload} className="download-btn">
              <Download size={16} />
              Download
            </button>
          )}
          {onClose && (
            <button onClick={onClose} className="close-btn">
              <X size={16} />
            </button>
          )}
        </div>
      </div>
      <div className="result-content">
        {resultImage ? (
          <div className="result-image">
            <img src={resultImage} alt="Processed result" />
          </div>
        ) : (
          <div className="result-placeholder">
            <p>Execute pipeline to see results</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default ResultViewer

