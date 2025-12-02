import { useEffect, useState } from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { CVModule } from '../types'
import './ModulePanel.css'

function ModulePanel() {
  const { modules, categories, loadModules, addNode } = usePipelineStore()
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    loadModules()
  }, [loadModules])

  const filteredModules = modules.filter(module => {
    const matchesCategory = !selectedCategory || module.category === selectedCategory
    const matchesSearch = !searchQuery || 
      module.display_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      module.description.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesCategory && matchesSearch
  })

  const handleAddModule = (module: CVModule) => {
    const nodeId = `node-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const position = {
      x: Math.random() * 400 + 100,
      y: Math.random() * 400 + 100,
    }

    addNode({
      id: nodeId,
      type: 'custom',
      position,
      data: {
        moduleId: module.id,
        moduleName: module.display_name,
        params: {},
      },
    })
  }

  const handleAddSpecialNode = (type: 'imageUpload' | 'result' | 'save', displayName: string) => {
    const nodeId = `${type}-${Date.now()}`
    const position = {
      x: Math.random() * 400 + 100,
      y: Math.random() * 400 + 100,
    }

    const baseData: any = {
      moduleName: displayName,
    }

    if (type === 'imageUpload') {
      baseData.imageData = null
      baseData.fileName = null
    } else if (type === 'result') {
      baseData.imageData = null
    } else if (type === 'save') {
      baseData.imageData = null
      baseData.fileName = null
    }

    addNode({
      id: nodeId,
      type,
      position,
      data: baseData,
    })
  }

  return (
    <div className="module-panel">
      <div className="panel-header">
        <h2>Modules</h2>
      </div>
      <div className="panel-search">
        <input
          type="text"
          placeholder="Search modules..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>
      <div className="panel-categories">
        <button
          className={!selectedCategory ? 'active' : ''}
          onClick={() => setSelectedCategory(null)}
        >
          All
        </button>
        {Object.keys(categories).map(cat => (
          <button
            key={cat}
            className={selectedCategory === cat ? 'active' : ''}
            onClick={() => setSelectedCategory(cat)}
          >
            {cat}
          </button>
        ))}
      </div>
      <div className="panel-content">
        {/* Special nodes section */}
        <div className="module-section">
          <h3 className="section-title">Input/Output</h3>
          <div className="module-item" onClick={() => handleAddSpecialNode('imageUpload', 'Image Upload')}>
            <h3>Image Upload</h3>
            <p>Upload an image to process</p>
            <span className="module-category">input</span>
          </div>
          <div className="module-item" onClick={() => handleAddSpecialNode('result', 'Result Viewer')}>
            <h3>Result Viewer</h3>
            <p>View processed image results</p>
            <span className="module-category">output</span>
          </div>
          <div className="module-item" onClick={() => handleAddSpecialNode('save', 'Save Image')}>
            <h3>Save Image</h3>
            <p>Download processed image</p>
            <span className="module-category">output</span>
          </div>
        </div>

        {/* CV Modules by category */}
        {Object.keys(categories).map(category => {
          const categoryModules = filteredModules.filter(m => m.category === category)
          if (categoryModules.length === 0 && selectedCategory !== category) return null
          
          return (
            <div key={category} className="module-section">
              <h3 className="section-title">{category}</h3>
              {categoryModules.map(module => (
                <div
                  key={module.id}
                  className="module-item"
                  onClick={() => handleAddModule(module)}
                >
                  <h3>{module.display_name}</h3>
                  <p>{module.description}</p>
                  <span className="module-category">{module.category}</span>
                </div>
              ))}
            </div>
          )
        })}
        
        {filteredModules.length === 0 && (
          <div className="no-modules">No modules found</div>
        )}
      </div>
    </div>
  )
}

export default ModulePanel

