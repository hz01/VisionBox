import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload } from 'lucide-react'
import './ImageUpload.css'

interface ImageUploadProps {
  onImageSelect: (image: string | null) => void
  selectedImage: string | null
}

function ImageUpload({ onImageSelect, selectedImage }: ImageUploadProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = () => {
        onImageSelect(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }, [onImageSelect])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 1,
  })

  return (
    <div className="image-upload">
      <div className="upload-header">
        <h2>Input Image</h2>
      </div>
      <div
        {...getRootProps()}
        className={`upload-area ${isDragActive ? 'active' : ''}`}
      >
        <input {...getInputProps()} />
        {selectedImage ? (
          <div className="image-preview">
            <img src={selectedImage} alt="Uploaded" />
            <button
              className="remove-image"
              onClick={(e) => {
                e.stopPropagation()
                onImageSelect(null)
              }}
            >
              Remove
            </button>
          </div>
        ) : (
          <div className="upload-placeholder">
            <Upload size={48} />
            <p>{isDragActive ? 'Drop image here' : 'Drag & drop or click to upload'}</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default ImageUpload

