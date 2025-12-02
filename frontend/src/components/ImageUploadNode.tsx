import { useCallback } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { useDropzone } from 'react-dropzone'
import { Upload, X } from 'lucide-react'
import { usePipelineStore } from '../store/pipelineStore'
import './ImageUploadNode.css'

function ImageUploadNode({ id, data, selected }: NodeProps) {
  const { updateNode } = usePipelineStore()

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = () => {
        updateNode(id, {
          data: {
            ...data,
            imageData: reader.result as string,
            fileName: file.name,
          },
        })
      }
      reader.readAsDataURL(file)
    }
  }, [id, data, updateNode])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 1,
  })

  const handleRemove = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    e.preventDefault()
    updateNode(id, {
      data: {
        ...data,
        imageData: null,
        fileName: null,
      },
    })
  }, [id, data, updateNode])

  const handleChangeImage = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    e.preventDefault()
    // Create a temporary input and trigger it
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/*'
    input.onchange = (event: any) => {
      const file = event.target.files?.[0]
      if (file) {
        onDrop([file])
      }
    }
    input.click()
  }, [onDrop])

  return (
    <div 
      className={`image-upload-node ${selected ? 'selected' : ''}`}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <div className="node-header">
        <h3>Image Upload</h3>
      </div>
      <div className="node-content" onMouseDown={(e) => e.stopPropagation()}>
        {data.imageData ? (
          <div className="image-preview-container">
            <img src={data.imageData} alt="Uploaded" className="preview-image" />
            <button className="remove-btn" onClick={handleRemove}>
              <X size={14} />
            </button>
            <div className="file-name">{data.fileName}</div>
            <button className="change-image-btn" onClick={handleChangeImage}>
              Change Image
            </button>
          </div>
        ) : (
          <div 
            {...getRootProps()} 
            className={`upload-area ${isDragActive ? 'active' : ''}`}
            onMouseDown={(e) => {
              // Stop ReactFlow from selecting the node, but allow dropzone to work
              e.stopPropagation()
            }}
          >
            <input {...getInputProps()} />
            <Upload size={24} />
            <p>{isDragActive ? 'Drop image here' : 'Click or drag image'}</p>
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}

export default ImageUploadNode

