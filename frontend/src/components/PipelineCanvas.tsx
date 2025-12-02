import { useCallback, useEffect, useRef, useMemo } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  NodeTypes,
  BackgroundVariant,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { usePipelineStore } from '../store/pipelineStore'
import CustomNode from './CustomNode'
import ImageUploadNode from './ImageUploadNode'
import ResultNode from './ResultNode'
import SaveNode from './SaveNode'
import ExecuteButton from './ExecuteButton'
import './PipelineCanvas.css'

interface PipelineCanvasProps {
  selectedImage: string | null
  onResultChange: (result: string | null) => void
  onOpenResultPanel: (image: string | null) => void
}

function PipelineCanvas({ selectedImage, onResultChange, onOpenResultPanel }: PipelineCanvasProps) {
  const {
    nodes: storeNodes,
    edges: storeEdges,
    setNodes,
    addEdge: addStoreEdge,
    removeNode,
    removeEdge,
  } = usePipelineStore()

  // Memoize nodeTypes to prevent recreation on every render
  const nodeTypes: NodeTypes = useMemo(() => ({
    custom: CustomNode,
    imageUpload: ImageUploadNode,
    result: (props: any) => <ResultNode {...props} onOpenResultPanel={onOpenResultPanel} />,
    save: SaveNode,
  }), [onOpenResultPanel])

  const [nodes, setNodesState, onNodesChange] = useNodesState(storeNodes)
  const [edges, setEdgesState, onEdgesChange] = useEdgesState(storeEdges)

  // Sync from store to ReactFlow when nodes change (including data changes)
  const lastStoreHashRef = useRef<string>('')
  useEffect(() => {
    // Create a hash of store nodes for comparison
    const storeHash = storeNodes.map(n => `${n.id}:${JSON.stringify(n.data)}`).sort().join('|')
    
    if (storeHash !== lastStoreHashRef.current) {
      lastStoreHashRef.current = storeHash
      setNodesState(storeNodes as any)
    }
  }, [storeNodes, setNodesState])

  useEffect(() => {
    const storeEdgeIds = storeEdges.map(e => e.id).sort().join(',')
    const currentEdgeIds = edges.map(e => e.id).sort().join(',')
    
    if (storeEdgeIds !== currentEdgeIds) {
      setEdgesState(storeEdges)
    }
  }, [storeEdges.map(e => e.id).join(','), edges, setEdgesState])

  // Sync position changes back to store (debounced)
  const positionSyncTimeout = useRef<ReturnType<typeof setTimeout>>()
  useEffect(() => {
    if (positionSyncTimeout.current) {
      clearTimeout(positionSyncTimeout.current)
    }
    positionSyncTimeout.current = setTimeout(() => {
      setNodes(nodes as any)
    }, 300)
    return () => {
      if (positionSyncTimeout.current) {
        clearTimeout(positionSyncTimeout.current)
      }
    }
  }, [nodes.map(n => `${n.id}-${Math.round(n.position.x)}-${Math.round(n.position.y)}`).join(',')])

  const onConnect = useCallback(
    (params: Connection) => {
      const edge = addEdge(params, edges)
      setEdgesState(edge)
      if (params.source && params.target) {
        addStoreEdge({
          id: `${params.source}-${params.target}`,
          source: params.source,
          target: params.target,
        })
      }
    },
    [edges, setEdgesState, addStoreEdge]
  )

  const onNodesDelete = useCallback(
    (deleted: Node[]) => {
      deleted.forEach(node => removeNode(node.id))
    },
    [removeNode]
  )

  const onEdgesDelete = useCallback(
    (deleted: Edge[]) => {
      deleted.forEach(edge => removeEdge(edge.id))
    },
    [removeEdge]
  )

  // Validate connections - prevent multiple inputs unless node allows it
  const { modules } = usePipelineStore()
  const isValidConnection = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) return false
      
      // Check if target already has an incoming connection
      const targetNode = nodes.find(n => n.id === connection.target)
      if (!targetNode) return false
      
      // Image upload nodes should not have inputs
      if (targetNode.type === 'imageUpload') return false
      
      // Generation nodes should not have inputs (they generate their own images)
      if (targetNode.type === 'custom' && targetNode.data.moduleId) {
        const module = modules.find(m => m.id === targetNode.data.moduleId)
        if (module && module.category === 'generation') {
          return false
        }
      }
      
      // Check if target already has incoming edges
      const hasIncomingEdge = edges.some(e => e.target === connection.target)
      
      // Check if the module allows multiple inputs
      let allowsMultiple = false
      if (targetNode.type === 'custom' && targetNode.data.moduleId) {
        const module = modules.find(m => m.id === targetNode.data.moduleId)
        if (module && module.allowMultipleInputs) {
          allowsMultiple = true
        }
      }
      
      // If node already has an input and doesn't allow multiple, reject
      if (hasIncomingEdge && !allowsMultiple) {
        return false
      }
      
      // Prevent self-connections
      if (connection.source === connection.target) {
        return false
      }
      
      return true
    },
    [nodes, edges, modules]
  )

  return (
    <div className="pipeline-canvas">
      <div className="canvas-header">
        <h2>Pipeline Canvas</h2>
        <ExecuteButton
          selectedImage={selectedImage}
          onResultChange={onResultChange}
        />
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodesDelete={onNodesDelete}
        onEdgesDelete={onEdgesDelete}
        isValidConnection={isValidConnection}
        nodeTypes={nodeTypes}
        nodesDraggable={true}
        nodesConnectable={true}
        elementsSelectable={true}
        selectNodesOnDrag={false}
        deleteKeyCode={['Backspace', 'Delete']}
        fitView
        className="react-flow-container"
      >
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  )
}

export default PipelineCanvas

