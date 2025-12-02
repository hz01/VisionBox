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
  getGenerationNode: () => PipelineNode | undefined
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
    
    // Filter out image upload and generation nodes (they're handled separately)
    const { modules } = get()
    const processingNodes = nodes.filter(n => {
      if (n.type === 'imageUpload') return false
      if (n.type === 'custom' && n.data.moduleId) {
        const module = modules.find((m: any) => m.id === n.data.moduleId)
        if (module && module.category === 'generation') return false
      }
      return true
    })
    
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

  getGenerationNode: () => {
    const { nodes, modules } = get()
    return nodes.find(n => {
      if (n.type === 'custom' && n.data.moduleId) {
        const module = modules.find((m: any) => m.id === n.data.moduleId)
        return module && module.category === 'generation'
      }
      return false
    })
  },

  getPathToNode: (targetNodeId: string) => {
    const { nodes, edges, modules } = get()
    
    // Find path from image upload node or generation node to target node
    const imageUploadNode = nodes.find(n => n.type === 'imageUpload' && n.data.imageData)
    const generationNode = nodes.find(n => {
      if (n.type === 'custom' && n.data.moduleId) {
        const module = modules.find(m => m.id === n.data.moduleId)
        return module && module.category === 'generation'
      }
      return false
    })
    
    const sourceNode = imageUploadNode || generationNode
    if (!sourceNode) return []
    
    // Build reverse graph (target -> source) to trace back
    const reverseEdges = new Map<string, string[]>()
    edges.forEach(e => {
      if (!reverseEdges.has(e.target)) {
        reverseEdges.set(e.target, [])
      }
      reverseEdges.get(e.target)!.push(e.source)
    })
    
    // BFS from target to find path to source node
    const visited = new Set<string>()
    const queue: { nodeId: string; path: string[] }[] = [{ nodeId: targetNodeId, path: [] }]
    
    while (queue.length > 0) {
      const { nodeId, path: currentPath } = queue.shift()!
      
      if (visited.has(nodeId)) continue
      visited.add(nodeId)
      
      if (nodeId === sourceNode.id) {
        // Found path, reverse it to get correct order (from source to target)
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
    const { nodes, edges, modules } = get()
    const steps: import('../types').PipelineStep[] = []
    
    // Check if first node is a generation node
    const firstNode = nodes.find(n => n.id === path[0])
    const isGenerationPath = firstNode && firstNode.type === 'custom' && firstNode.data.moduleId && 
      modules.find((m: any) => m.id === firstNode.data.moduleId && m.category === 'generation')
    
    // Start from index 0 if it's a generation path, otherwise skip image upload (index 0)
    const startIndex = isGenerationPath ? 0 : 1
    
    // Include all processing nodes, including result nodes if they have outputs
    for (let i = startIndex; i < path.length; i++) {
      const nodeId = path[i]
      const node = nodes.find(n => n.id === nodeId)
      
      if (!node) continue
      
      // Skip save nodes (they're endpoints)
      if (node.type === 'save') break
      
      // Skip image upload nodes (they're sources, not processing steps)
      if (node.type === 'imageUpload') continue
      
      // Include result nodes only if they have outgoing edges (pass-through)
      if (node.type === 'result') {
        const hasOutput = edges.some(e => e.source === nodeId)
        if (!hasOutput) {
          // Result node without output is an endpoint
          break
        }
        // Result node with output passes through, continue to next node
        continue
      }
      
      // Include processing nodes with moduleId (including generation nodes)
      if (node.data.moduleId) {
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

