import { create } from 'zustand'
import { PipelineNode, PipelineEdge, CVModule } from '../types'
import { moduleApi } from '../services/api'

interface PipelineStore {
  nodes: PipelineNode[]
  edges: PipelineEdge[]
  modules: CVModule[]
  categories: Record<string, string[]>
  selectedNodeId: string | null
  isLoading: boolean
  
  // Actions
  setNodes: (nodes: PipelineNode[]) => void
  setEdges: (edges: PipelineEdge[]) => void
  addNode: (node: PipelineNode) => void
  removeNode: (nodeId: string) => void
  updateNode: (nodeId: string, updates: Partial<PipelineNode>) => void
  addEdge: (edge: PipelineEdge) => void
  removeEdge: (edgeId: string) => void
  setSelectedNodeId: (id: string | null) => void
  loadModules: () => Promise<void>
  getPipelineSteps: () => import('../types').PipelineStep[]
  getImageUploadNode: () => PipelineNode | undefined
  getPathToNode: (targetNodeId: string) => string[]
  getPipelineStepsForPath: (path: string[]) => import('../types').PipelineStep[]
}

export const usePipelineStore = create<PipelineStore>((set, get) => ({
  nodes: [],
  edges: [],
  modules: [],
  categories: {},
  selectedNodeId: null,
  isLoading: false,

  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),

  addNode: (node) => set((state) => ({ nodes: [...state.nodes, node] })),
  
  removeNode: (nodeId) => set((state) => ({
    nodes: state.nodes.filter(n => n.id !== nodeId),
    edges: state.edges.filter(e => e.source !== nodeId && e.target !== nodeId),
    selectedNodeId: state.selectedNodeId === nodeId ? null : state.selectedNodeId,
  })),

  updateNode: (nodeId, updates) => set((state) => ({
    nodes: state.nodes.map(n => n.id === nodeId ? { ...n, ...updates } : n),
  })),

  addEdge: (edge) => set((state) => {
    // Check if edge already exists
    const exists = state.edges.some(
      e => e.source === edge.source && e.target === edge.target
    )
    if (exists) return state
    return { edges: [...state.edges, edge] }
  }),

  removeEdge: (edgeId) => set((state) => ({
    edges: state.edges.filter(e => e.id !== edgeId),
  })),

  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  loadModules: async () => {
    set({ isLoading: true })
    try {
      const data = await moduleApi.getAll()
      set({ modules: data.modules, categories: data.categories, isLoading: false })
    } catch (error) {
      console.error('Failed to load modules:', error)
      set({ isLoading: false })
    }
  },

  getPipelineSteps: () => {
    const { nodes, edges } = get()
    
    // Filter out image upload nodes (they're handled separately)
    const processingNodes = nodes.filter(n => n.type !== 'imageUpload')
    
    // Build dependency graph to determine execution order
    const nodeMap = new Map(processingNodes.map(n => [n.id, n]))
    const incoming = new Map<string, string[]>()
    const outgoing = new Map<string, string[]>()
    
    processingNodes.forEach(n => {
      incoming.set(n.id, [])
      outgoing.set(n.id, [])
    })
    
    edges.forEach(e => {
      if (nodeMap.has(e.source) && nodeMap.has(e.target)) {
        incoming.get(e.target)?.push(e.source)
        outgoing.get(e.source)?.push(e.target)
      }
    })
    
    // Topological sort
    const visited = new Set<string>()
    const result: import('../types').PipelineStep[] = []
    
    const visit = (nodeId: string) => {
      if (visited.has(nodeId)) return
      visited.add(nodeId)
      
      const node = nodeMap.get(nodeId)!
      if (node.data.moduleId) {
        result.push({
          id: node.id,
          module: node.data.moduleId,
          params: node.data.params || {},
        })
      }
      
      // Visit dependents
      outgoing.get(nodeId)?.forEach(visit)
    }
    
    // Start with nodes that have no dependencies
    processingNodes.forEach(n => {
      if (incoming.get(n.id)?.length === 0) {
        visit(n.id)
      }
    })
    
    // If there are cycles or unvisited nodes, just use node order
    if (result.length !== processingNodes.length) {
      return processingNodes
        .filter(n => n.data.moduleId)
        .map(n => ({
          id: n.id,
          module: n.data.moduleId!,
          params: n.data.params || {},
        }))
    }
    
    return result
  },

  getImageUploadNode: () => {
    const { nodes } = get()
    return nodes.find(n => n.type === 'imageUpload' && n.data.imageData)
  },

  getPathToNode: (targetNodeId: string) => {
    const { nodes, edges } = get()
    
    // Find path from image upload node to target node
    const imageUploadNode = nodes.find(n => n.type === 'imageUpload' && n.data.imageData)
    if (!imageUploadNode) return []
    
    // Build reverse graph (target -> source) to trace back
    const reverseEdges = new Map<string, string[]>()
    edges.forEach(e => {
      if (!reverseEdges.has(e.target)) {
        reverseEdges.set(e.target, [])
      }
      reverseEdges.get(e.target)!.push(e.source)
    })
    
    // BFS from target to find path to image upload
    const visited = new Set<string>()
    const queue: { nodeId: string; path: string[] }[] = [{ nodeId: targetNodeId, path: [] }]
    
    while (queue.length > 0) {
      const { nodeId, path: currentPath } = queue.shift()!
      
      if (visited.has(nodeId)) continue
      visited.add(nodeId)
      
      if (nodeId === imageUploadNode.id) {
        // Found path, reverse it to get correct order (from upload to target)
        return [...currentPath, nodeId].reverse()
      }
      
      const sources = reverseEdges.get(nodeId) || []
      for (const sourceId of sources) {
        const sourceNode = nodes.find(n => n.id === sourceId)
        if (sourceNode && !visited.has(sourceId)) {
          queue.push({ 
            nodeId: sourceId, 
            path: [...currentPath, nodeId] 
          })
        }
      }
    }
    
    return []
  },

  getPipelineStepsForPath: (path: string[]) => {
    const { nodes } = get()
    const steps: import('../types').PipelineStep[] = []
    
    // Skip first (image upload) and last (result/save) nodes
    for (let i = 1; i < path.length - 1; i++) {
      const nodeId = path[i]
      const node = nodes.find(n => n.id === nodeId)
      if (node && node.data.moduleId) {
        steps.push({
          id: node.id,
          module: node.data.moduleId,
          params: node.data.params || {},
        })
      }
    }
    
    return steps
  },
}))

