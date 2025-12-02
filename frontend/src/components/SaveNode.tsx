import { memo } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { Download } from 'lucide-react'
import './SaveNode.css'

function SaveNode({ id, data, selected }: NodeProps) {
  const handleSave = () => {
    if (data.imageData) {
      const link = document.createElement('a')
      link.href = data.imageData
      link.download = data.fileName || `visionbox-result-${Date.now()}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  return (
    <div className={`save-node ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Top} />
      <div className="node-header">
        <h3>Save Image</h3>
      </div>
      <div className="node-content">
        <button 
          className="save-button" 
          onClick={handleSave}
          disabled={!data.imageData}
        >
          <Download size={16} />
          Save Image
        </button>
        {data.imageData && (
          <p className="save-hint">Ready to save</p>
        )}
      </div>
    </div>
  )
}

export default SaveNode

