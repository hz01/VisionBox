import { useState } from 'react'
import { Play, Loader } from 'lucide-react'
import { usePipelineStore } from '../store/pipelineStore'
import { pipelineApi } from '../services/api'
import './ExecuteButton.css'

interface ExecuteButtonProps {
  selectedImage: string | null
  getPipelineSteps: () => import('../types').PipelineStep[]
  onResultChange: (result: string | null) => void
  onOpenResultPanel?: (image: string | null) => void
}

function ExecuteButton({ selectedImage, getPipelineSteps, onResultChange, onOpenResultPanel }: ExecuteButtonProps) {
  const [isExecuting, setIsExecuting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { getImageUploadNode, nodes, updateNode, getPathToNode, getPipelineStepsForPath } = usePipelineStore()

  const handleExecute = async () => {
    // Get image from upload node or fallback to selectedImage prop
    const imageUploadNode = getImageUploadNode()
    const imageData = imageUploadNode?.data.imageData || selectedImage

    if (!imageData) {
      setError('Please upload an image first (add Image Upload node)')
      return
    }

    setIsExecuting(true)
    setError(null)

    try {
      // Convert data URL to File
      const base64Data = imageData.split(',')[1]
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'image/png' })
      const file = new File([blob], 'image.png', { type: 'image/png' })

      // Get all result and save nodes
      const resultNodes = nodes.filter(n => n.type === 'result' || n.type === 'save')
      
      if (resultNodes.length === 0) {
        setError('Please add at least one Result Viewer or Save node')
        setIsExecuting(false)
        return
      }

      // Execute pipeline for each result/save node separately
      const executionPromises = resultNodes.map(async (node) => {
        // Find path from image upload to this node
        const path = getPathToNode(node.id)
        
        if (path.length === 0) {
          // No path found, skip this node
          return { nodeId: node.id, success: false, error: 'No path from image upload to this node' }
        }

        // Get pipeline steps for this path
        const pipeline = getPipelineStepsForPath(path)
        
        if (pipeline.length === 0 && path.length === 2) {
          // Direct connection from image upload to result (no processing)
          // Just use the original image
          return { 
            nodeId: node.id, 
            success: true, 
            image_base64: imageData 
          }
        }

        try {
          const result = await pipelineApi.execute(file, pipeline)
          return {
            nodeId: node.id,
            success: result.success,
            image_base64: result.image_base64,
            errors: result.errors
          }
        } catch (err: any) {
          return {
            nodeId: node.id,
            success: false,
            error: err.message || 'Execution failed'
          }
        }
      })

      const results = await Promise.all(executionPromises)
      
      // Update each node with its specific result
      let hasSuccess = false
      const errors: string[] = []
      
      results.forEach((result) => {
        if (result.success && result.image_base64) {
          hasSuccess = true
          const node = nodes.find(n => n.id === result.nodeId)
          if (node) {
            updateNode(result.nodeId, {
              data: {
                ...node.data,
                imageData: result.image_base64,
                fileName: node.type === 'save' ? `visionbox-result-${Date.now()}.png` : node.data.fileName,
              },
            })
          }
        } else {
          errors.push(`${result.nodeId}: ${result.error || result.errors?.join(', ') || 'Failed'}`)
        }
      })

      if (hasSuccess) {
        // Use the first successful result for the main result panel
        const firstSuccess = results.find(r => r.success && r.image_base64)
        if (firstSuccess?.image_base64) {
          onResultChange(firstSuccess.image_base64)
        }
      }

      if (errors.length > 0) {
        setError(`Some pipelines failed: ${errors.join('; ')}`)
      } else if (!hasSuccess) {
        setError('All pipeline executions failed')
      }
    } catch (err: any) {
      setError(err.message || 'Failed to execute pipeline')
      console.error('Pipeline execution error:', err)
    } finally {
      setIsExecuting(false)
    }
  }

  return (
    <div className="execute-button-container">
      {error && <div className="error-message">{error}</div>}
      <button
        onClick={handleExecute}
        disabled={isExecuting}
        className="execute-button"
      >
        {isExecuting ? (
          <>
            <Loader className="spinner" size={16} />
            Executing...
          </>
        ) : (
          <>
            <Play size={16} />
            Execute Pipeline
          </>
        )}
      </button>
    </div>
  )
}

export default ExecuteButton

