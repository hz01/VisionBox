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

  const renderInput = (param: ModuleParameter) => {
    const value = params[param.name] ?? param.default

    switch (param.type) {
      case 'int':
      case 'float':
        return (
          <div key={param.name} className="param-input">
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
          <div key={param.name} className="param-input">
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
          <div key={param.name} className="param-input">
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
          <div key={param.name} className="param-input">
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

  if (module.length === 0) {
    return <div className="no-params">No parameters</div>
  }

  return (
    <div className="node-settings">
      {module.map(renderInput)}
    </div>
  )
}

export default NodeSettings

