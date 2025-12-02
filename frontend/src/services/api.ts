import axios from 'axios'
import { CVModule, PipelineStep } from '../types'

const API_BASE_URL = '/api/v1'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const moduleApi = {
  getAll: async (): Promise<{ modules: CVModule[]; categories: Record<string, string[]> }> => {
    const response = await api.get('/modules')
    return response.data
  },

  getById: async (moduleId: string): Promise<CVModule> => {
    const response = await api.get(`/modules/${moduleId}`)
    return response.data
  },
}

export const pipelineApi = {
  validate: async (pipeline: PipelineStep[]): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> => {
    const response = await api.post('/pipeline/validate', { pipeline })
    return response.data
  },

  execute: async (imageFile: File, pipeline: PipelineStep[]): Promise<{ success: boolean; image_base64: string | null; errors: string[] }> => {
    const formData = new FormData()
    formData.append('file', imageFile)
    formData.append('pipeline', JSON.stringify(pipeline))

    try {
      const response = await api.post('/pipeline/execute', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      return response.data
    } catch (error: any) {
      // Extract error message from response
      if (error.response) {
        const errorMessage = error.response.data?.detail || error.response.data?.message || 'Unknown error'
        throw new Error(errorMessage)
      }
      throw error
    }
  },
}

