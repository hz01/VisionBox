import { useEffect, useState } from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { ModuleParameter } from '../types'
import './NodeSettings.css'

interface NodeSettingsProps {
  moduleId: string
  params: Record<string, any>
  onParamChange: (paramName: string, value: any) => void
}

function NodeSettings({ moduleId, params, onParamChange }: NodeSettingsProps) {
  const { modules } = usePipelineStore()
  const [module, setModule] = useState<ModuleParameter[]>([])

  useEffect(() => {
    const found = modules.find(m => m.id === moduleId)
    if (found) {
      setModule(found.parameters)
    } else {
      setModule([])
    }
  }, [moduleId, modules])

  // Filter parameters based on selected model (for face detection, GAN, and similar modules)
  const getFilteredParameters = () => {
    const selectedModel = params.model?.toString() || params.model
    
    // If no model is selected or module doesn't have model parameter, show all
    const hasModelParam = module.some(p => p.name === 'model')
    if (!hasModelParam || !selectedModel) {
      return module
    }

    // Special handling for model name to parameter prefix mapping
    const modelToPrefix: Record<string, string> = {
      'noise_pattern': 'noise_',
      'progressive_gan': 'progressive_gan_',
    }
    
    const modelPrefix = modelToPrefix[selectedModel.toLowerCase()] || (selectedModel.toLowerCase() + '_')
    const modelPrefixes = [
      'haar_', 'mediapipe_', 'dlib_', 'yolo_', 'mtcnn_', 'retinaface_',
      'dcgan_', 'stylegan_', 'stylegan3_', 'progressive_gan_', 
      'biggan_', 'wgan_', 'noise_', 'kmeans_',
      'realesrgan_', 'swinir_', 'esrgan_', 'edsr_', 'rcan_',
      'sd_', 'codeformer_', 'fsrcnn_', 'espcn_'
    ]

    return module.filter(param => {
      // Always show the model selector
      if (param.name === 'model') {
        return true
      }
      
      // Check if this param has a model prefix
      const hasModelPrefix = modelPrefixes.some(prefix => param.name.startsWith(prefix))
      
      if (!hasModelPrefix) {
        // Common parameter without model prefix, always show
        // For GAN: width, height, seed, color_mode are common
        return true
      }
      
      // Show parameters that match the selected model prefix
      return param.name.startsWith(modelPrefix)
    })
  }

  const renderInput = (param: ModuleParameter) => {
    const value = params[param.name] ?? param.default
    
    // Check if this is a generation module
    const foundModule = modules.find(m => m.id === moduleId)
    const isGenerationModule = foundModule?.category === 'generation'
    
    // Model selector and some important params should span full width in 2-column layout
    const shouldSpanFullWidth = isGenerationModule && (
      param.name === 'model' || 
      param.name === 'width' || 
      param.name === 'height' ||
      param.name === 'seed' ||
      param.name === 'color_mode'
    )

    switch (param.type) {
      case 'int':
      case 'float':
        return (
          <div key={param.name} className={`param-input ${shouldSpanFullWidth ? 'full-width' : ''}`}>
            <label>{param.name}</label>
            <input
              type="number"
              value={value}
              min={param.min}
              max={param.max}
              step={param.type === 'float' ? 0.1 : 1}
              onChange={(e) => onParamChange(param.name, param.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value))}
            />
            {param.description && <span className="param-hint">{param.description}</span>}
          </div>
        )

      case 'bool':
        return (
          <div key={param.name} className={`param-input ${shouldSpanFullWidth ? 'full-width' : ''}`}>
            <label>
              <input
                type="checkbox"
                checked={value}
                onChange={(e) => onParamChange(param.name, e.target.checked)}
              />
              {param.name}
            </label>
            {param.description && <span className="param-hint">{param.description}</span>}
          </div>
        )

      case 'select':
        return (
          <div key={param.name} className={`param-input ${shouldSpanFullWidth ? 'full-width' : ''}`}>
            <label>{param.name}</label>
            <select
              value={value}
              onChange={(e) => onParamChange(param.name, e.target.value)}
            >
              {param.options?.map(opt => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
            {param.description && <span className="param-hint">{param.description}</span>}
          </div>
        )

      default:
        return (
          <div key={param.name} className={`param-input ${shouldSpanFullWidth ? 'full-width' : ''}`}>
            <label>{param.name}</label>
            <input
              type="text"
              value={value}
              onChange={(e) => onParamChange(param.name, e.target.value)}
            />
            {param.description && <span className="param-hint">{param.description}</span>}
          </div>
        )
    }
  }

  const filteredParams = getFilteredParameters()

  if (filteredParams.length === 0) {
    return <div className="no-params">No parameters</div>
  }

  // Check if this is a generation module (GAN) - use 2 columns
  const foundModule = modules.find(m => m.id === moduleId)
  const isGenerationModule = foundModule?.category === 'generation'

  return (
    <div className={`node-settings ${isGenerationModule ? 'two-columns' : ''}`}>
      {filteredParams.map(renderInput)}
    </div>
  )
}

export default NodeSettings

