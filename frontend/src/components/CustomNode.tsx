import { useCallback, useEffect, useRef } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { usePipelineStore } from '../store/pipelineStore'
import NodeSettings from './NodeSettings'
import './CustomNode.css'

function CustomNode({ id, data, selected }: NodeProps) {
  const { updateNode, modules } = usePipelineStore()

  // Initialize params when module is first loaded (only once)
  const initKey = `${id}-${data.moduleId}`
  const hasInitialized = useRef<string>('')
  
  useEffect(() => {
    if (!data.moduleId || hasInitialized.current === initKey) return
    const found = modules.find(m => m.id === data.moduleId)
    if (found && found.parameters) {
      const currentParams = data.params || {}
      const needsInit = found.parameters.some(
        param => !(param.name in currentParams) && param.default !== undefined
      )
      if (needsInit) {
        hasInitialized.current = initKey
        const newParams: Record<string, any> = { ...currentParams }
        found.parameters.forEach(param => {
          if (!(param.name in newParams) && param.default !== undefined) {
            newParams[param.name] = param.default
          }
        })
        // Use setTimeout to avoid state update during render
        setTimeout(() => {
          updateNode(id, {
            data: {
              ...data,
              params: newParams,
            },
          })
        }, 0)
      }
    }
  }, [data.moduleId]) // Only when moduleId changes

  const handleParamChange = useCallback((paramName: string, value: any) => {
    updateNode(id, {
      data: {
        ...data,
        params: {
          ...data.params || {},
          [paramName]: value,
        },
      },
    })
  }, [id, data, updateNode])

  const handleContentClick = useCallback((e: React.MouseEvent) => {
    // Prevent node selection when clicking on content, but allow input interactions
    const target = e.target as HTMLElement
    if (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'LABEL') {
      return // Allow input interactions
    }
    e.stopPropagation()
  }, [])

  // Check if this is a generation module (doesn't need input)
  const module = modules.find(m => m.id === data.moduleId)
  const isGenerationModule = module?.category === 'generation'

  return (
    <div 
      className={`custom-node ${selected ? 'selected' : ''}`}
      onMouseDown={(e) => {
        // Allow dragging from header, but prevent selection on content click
        const target = e.target as HTMLElement
        if (target.closest('.node-content')) {
          e.stopPropagation()
        }
      }}
    >
      {!isGenerationModule && <Handle type="target" position={Position.Top} />}
      <div className="node-header">
        <h3>{data.moduleName}</h3>
      </div>
      <div className="node-content" onClick={handleContentClick} onMouseDown={(e) => e.stopPropagation()}>
        <NodeSettings
          moduleId={data.moduleId}
          params={data.params || {}}
          onParamChange={handleParamChange}
        />
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}

export default CustomNode

