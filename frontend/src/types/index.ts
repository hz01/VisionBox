export interface CVModule {
  id: string
  display_name: string
  description: string
  category: string
  parameters: ModuleParameter[]
  allowMultipleInputs?: boolean // If true, node can accept multiple input connections
}

export interface ModuleParameter {
  name: string
  type: 'int' | 'float' | 'str' | 'bool' | 'select'
  default: any
  min?: number
  max?: number
  options?: string[]
  description: string
}

export interface PipelineNode {
  id: string
  type: string
  position: { x: number; y: number }
  data: {
    moduleId?: string
    moduleName?: string
    params?: Record<string, any>
    // For image upload nodes
    imageData?: string | null
    fileName?: string | null
  }
}

export interface PipelineEdge {
  id: string
  source: string
  target: string
}

export interface PipelineStep {
  id: string
  module: string
  params: Record<string, any>
}

