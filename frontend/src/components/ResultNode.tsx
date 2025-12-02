import { Handle, Position, NodeProps } from 'reactflow'
import './ResultNode.css'

interface ResultNodeProps extends NodeProps {
  onOpenResultPanel?: (image: string | null) => void
}

function ResultNode({ id, data, selected, onOpenResultPanel }: ResultNodeProps) {
  const handleClick = () => {
    if (data.imageData && onOpenResultPanel) {
      onOpenResultPanel(data.imageData)
    }
  }

  return (
    <div className={`result-node ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Top} />
      <div className="node-header">
        <h3>Result Viewer</h3>
      </div>
      <div className="node-content" onClick={handleClick} style={{ cursor: data.imageData ? 'pointer' : 'default' }}>
        {data.imageData ? (
          <div className="result-preview">
            <img src={data.imageData} alt="Result" />
            <div className="click-hint">Click to view in panel</div>
          </div>
        ) : (
          <div className="no-result">
            <p>Connect to a processing node to see results</p>
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}

export default ResultNode

