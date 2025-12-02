import { useState } from 'react'
import { Play, Loader } from 'lucide-react'
import { usePipelineStore } from '../store/pipelineStore'
import { pipelineApi } from '../services/api'
import './ExecuteButton.css'

interface ExecuteButtonProps {
  selectedImage: string | null
  onResultChange: (result: string | null) => void
}

function ExecuteButton({ selectedImage, onResultChange }: ExecuteButtonProps) {
  const [isExecuting, setIsExecuting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { getImageUploadNode, getGenerationNode, nodes, edges, updateNode, getPathToNode, getPipelineStepsForPath } = usePipelineStore()

  const handleExecute = async () => {
    // Get image from upload node, generation node, or fallback to selectedImage prop
    const imageUploadNode = getImageUploadNode()
    const generationNode = getGenerationNode()
    
    // For generation nodes, we'll generate the image during execution
    // For now, check if we have an image source
    const imageData = imageUploadNode?.data.imageData || selectedImage

    if (!imageData && !generationNode) {
      setError('Please upload an image (add Image Upload node) or add a Generation node')
      return
    }

    setIsExecuting(true)
    setError(null)

    try {
      // Check if we have a source (image upload or generation node)
      if (!imageData && !generationNode) {
        setError('Please upload an image (add Image Upload node) or add a Generation node')
        setIsExecuting(false)
        return
      }
      
      // Convert image data to File if we have it (for non-generation pipelines)
      let file: File | null = null
      if (imageData) {
        const base64Data = imageData.split(',')[1]
        const byteCharacters = atob(base64Data)
        const byteNumbers = new Array(byteCharacters.length)
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i)
        }
        const byteArray = new Uint8Array(byteNumbers)
        const blob = new Blob([byteArray], { type: 'image/png' })
        file = new File([blob], 'image.png', { type: 'image/png' })
      }

      // Get all result and save nodes (endpoints)
      // An endpoint is a save node or a result node with no outgoing edges
      const endpointNodes = nodes.filter(n => {
        if (n.type === 'save') return true
        if (n.type === 'result') {
          // Result node is an endpoint if it has no outgoing edges
          const hasOutput = edges.some(e => e.source === n.id)
          return !hasOutput
        }
        return false
      })
      
      if (endpointNodes.length === 0) {
        setError('Please add at least one Result Viewer or Save node')
        setIsExecuting(false)
        return
      }

      // Execute pipeline for each endpoint node separately
      const executionPromises = endpointNodes.map(async (node) => {
        // Find path from image upload to this node
        const path = getPathToNode(node.id)
        
        if (path.length === 0) {
          // No path found, skip this node
          return { nodeId: node.id, success: false, error: 'No path from image upload to this node' }
        }

        // Get pipeline steps for this path (includes nodes up to and through result nodes)
        const pipeline = getPipelineStepsForPath(path)
        
        // Check if this is a generation pipeline
        const isGenerationPipeline = pipeline.length > 0 && pipeline[0].module === 'gan_generation'
        
        if (pipeline.length === 0 && path.length === 2 && imageData) {
          // Direct connection from image upload to result (no processing)
          // Just use the original image
          return { 
            nodeId: node.id, 
            success: true, 
            image_base64: imageData 
          }
        }
        
        if (pipeline.length === 0) {
          return { nodeId: node.id, success: false, error: 'No processing steps in pipeline' }
        }

        try {
          // For generation pipelines, file can be null
          const result = await pipelineApi.execute(isGenerationPipeline ? null : file, pipeline)
          
          // Update all result nodes in the path (not just the endpoint)
          const resultNodesInPath = path
            .map(nodeId => nodes.find(n => n.id === nodeId))
            .filter(n => n && n.type === 'result')
          
          // Update each result node in the path
          resultNodesInPath.forEach(resultNode => {
            if (resultNode) {
              updateNode(resultNode.id, {
                data: {
                  ...resultNode.data,
                  imageData: result.image_base64,
                },
              })
            }
          })
          
          return {
            nodeId: node.id,
            success: result.success,
            image_base64: result.image_base64,
            errors: result.errors,
            updatedNodes: resultNodesInPath.map(n => n?.id).filter(Boolean) as string[]
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

