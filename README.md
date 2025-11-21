  # 🎥 YOLO Video Object Detection with Distance Estimation
  
  A real-time object detection system that analyzes videos, identifies objects, and estimates their distance from the camera. Built with YOLOv8, FastAPI, and Streamlit.
  
  ## ✨ Features
  
  ### 🎯 Core Capabilities
  - **Real-time Object Detection**: Detects 80+ object classes using YOLOv8
  - **Distance Estimation**: Calculates object distance in meters using computer vision
  - **Position Tracking**: Identifies object position (Left, Center, Right)
  - **Confidence Scoring**: Shows detection confidence for each object
  
  ### 🎨 User Interface
  - **Beautiful Dashboard**: Interactive Streamlit frontend with gradient cards
  - **Live Detection Feed**: Real-time display with icons and arrows
  - **Color-Coded Alerts**: 
    - 🔴 Very Close (< 5m)
    - 🟡 Close (5-15m)
    - 🟢 Far (> 15m)
  
  ### 📊 Analytics & Reports
  - **Interactive Charts**: Bar charts and pie charts for detection distribution
  - **Multiple Export Formats**: 
    - CSV (for Excel/data analysis)
    - JSON (for API integration)
    - TXT (for reports)
  - **Detailed Metadata**: Frame-by-frame detection data with timestamps
  
  ### 🚀 Performance
  - **Fast Processing**: Multi-threaded video processing
  - **Multiple Codecs**: Support for H.264, MPEG-4, XVID
  - **Large File Support**: Handles videos up to 500MB
  - **Format Flexibility**: MP4, AVI, MOV, MKV, FLV, WMV
  
  ## 🎬 Demo
  
  ### Detection Output
  ```
  ┌─────────────────────────────────────────────────┐
  │ DETECTED OBJECTS              [Total: 8]        │
  ├─────────────────────────────────────────────────┤
  │ • Car - 15.5m away (Right)                      │
  │ • Person - 8.2m away (Left)                     │
  │ • Truck - 22.3m away (Center)                   │
  │ • Bicycle - 5.1m away (Left)                    │
  │ • Dog - 3.8m away (Right)                       │
  │ • Bus - 30.0m away (Center)                     │
  │ • Motorcycle - 12.0m away (Right)               │
  │ • Tree (Left)                                   │
  └─────────────────────────────────────────────────┘
  ```
  
  ## 🛠️ Installation
  
  ### Prerequisites
  - Python 3.8 or higher
  - pip (Python package manager)
  - CUDA (optional, for GPU acceleration)
  
  ### Step 1: Clone the Repository
  ```bash
  git clone https://github.com/yourusername/yolo-video-detection.git
  cd yolo-video-detection
  ```
  
  ### Step 2: Create Virtual Environment
  ```bash
  # Windows
  python -m venv venv
  venv\Scripts\activate
  
  # Linux/Mac
  python3 -m venv venv
  source venv/bin/activate
  ```
  
  ### Step 3: Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```
  
  ### Step 4: Download YOLO Model
  ```bash
  # Create models directory
  mkdir models
  
  # Download YOLOv8 model (or use your custom trained model)
  # Place your model file as: models/best.pt
  ```
  
  ### Step 5: Create Required Folders
  ```bash
  mkdir uploads
  mkdir jobs
  ```
  
  ## 🚀 Usage
  
  ### Starting the Backend (FastAPI)
  ```bash
  python main.py
  ```
  The API will be available at: `http://localhost:8000`
  
  ### Starting the Frontend (Streamlit)
  Open a new terminal and run:
  ```bash
  streamlit run frontend.py
  ```
  The web interface will open at: `http://localhost:8501`
  
  ### Using the Web Interface
  1. **Upload Video**: Click "Choose a video file" and select your video
  2. **Preview**: Check the original video preview
  3. **Process**: Click "🚀 Process Video" button
  4. **Wait**: Processing time depends on video length
  5. **View Results**: Check the "Results & Analysis" tab
  6. **Download**: Get the processed video and detection reports
  
  ### Using the API Directly
  
  #### Upload and Process Video
  ```bash
  curl -X POST "http://localhost:8000/upload" \
    -F "video=@path/to/your/video.mp4"
  ```
  
  #### Download Processed Video
  ```bash
  curl -X GET "http://localhost:8000/download/detected_filename.mp4" \
    --output processed_video.mp4
  ```
  
  #### Get Detection Metadata
  ```bash
  curl -X GET "http://localhost:8000/metadata/job_id"
  ```
  
  ## 📁 Project Structure
  
  ```
  yolo-video-detection/
  │
  ├── main.py                 # FastAPI backend server
  ├── frontend.py             # Streamlit frontend interface
  ├── requirements.txt        # Python dependencies
  ├── README.md              # This file
  │
  ├── models/
  │   └── best.pt            # YOLOv8 model file
  │
  ├── uploads/               # Temporary uploaded videos
  ├── jobs/                  # Processed videos and metadata
  │
  └── README.md
  ```
  
  ## 🔧 How It Works
  
  ### 1. Video Upload
  - User uploads video through Streamlit interface
  - Video is sent to FastAPI backend
  - File validation (format, size) is performed
  
  ### 2. Object Detection
  ```python
  # YOLO inference on each frame
  results = model(frame, conf=0.3)
  
  # Process each detection
  for box in results.boxes:
      - Extract bounding box coordinates
      - Get object class and confidence
      - Calculate object height in pixels
  ```
  
  ### 3. Distance Estimation
  ```python
  # Formula: Distance = (Real_Height × Focal_Length) / Pixel_Height
  distance = (REAL_HEIGHT[class] * FOCAL_LENGTH) / pixel_height
  
  # Example:
  # Person (1.7m tall) appears as 200 pixels
  # Distance = (1.7 × 700) / 200 = 5.95 meters
  ```
  
  ### 4. Position Detection
  ```python
  # Divide frame into three zones
  if x_center < width * 0.33:
      position = "Left"
  elif x_center > width * 0.66:
      position = "Right"
  else:
      position = "Center"
  ```
  
  ### 5. Visualization
  - Bounding boxes drawn on detected objects
  - Labels showing class, confidence, and distance
  - Information panel at top of frame
  - Color-coded text for better visibility
  
  ### 6. Metadata Generation
  - Frame-by-frame detection data
  - Object counts and distribution
  - Statistical analysis
  - Export in multiple formats
  
  ## 📡 API Documentation
  
  ### Base URL
  ```
  http://localhost:8000
  ```
  
  ### Endpoints
  
  #### 1. Health Check
  ```http
  GET /health
  ```
  **Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_path": "models/best.pt"
  }
  ```
  
  #### 2. Upload Video
  ```http
  POST /upload
  ```
  **Parameters:**
  - `video`: Video file (multipart/form-data)
  
  **Response:**
  ```json
  {
    "message": "Video processed successfully",
    "job_id": "1234567890_abc123",
    "output_filename": "detected_video.mp4",
    "download_url": "/download/detected_video.mp4",
    "metadata_url": "/metadata/1234567890_abc123",
    "total_detections": 156,
    "detection_summary": {
      "car": 45,
      "person": 32,
      "truck": 12
    }
  }
  ```
  
  #### 3. Download Processed Video
  ```http
  GET /download/{filename}
  ```
  
  #### 4. Get Metadata
  ```http
  GET /metadata/{job_id}
  ```
  **Response:**
  ```json
  {
    "job_id": "1234567890_abc123",
    "video_info": {
      "width": 1920,
      "height": 1080,
      "fps": 30,
      "total_frames": 900,
      "duration_seconds": 30
    },
    "detection_summary": {
      "car": 45,
      "person": 32
    },
    "total_detections": 156,
    "detections": [...]
  }
  ```
  
  #### 5. Cleanup Old Files
  ```http
  DELETE /cleanup
  ```
  
  ## ⚙️ Configuration
  
  ### Distance Estimation Parameters
  
  Edit `main.py` to adjust these values:
  
  ```python
  # Real-world object heights (in meters)
  REAL_HEIGHT = {
      "person": 1.7,
      "car": 1.4,
      "truck": 3.0,
      "bus": 3.2,
      "motorcycle": 1.2,
      "bicycle": 1.2,
      "dog": 0.6,
      "cat": 0.3,
      # Add more classes...
  }
  
  # Camera focal length (adjust based on your camera)
  FOCAL_LENGTH = 700
  ```
  
  ### File Upload Limits
  
  ```python
  # Maximum file size (500MB)
  MAX_FILE_SIZE = 500 * 1024 * 1024
  
  # Allowed video formats
  ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
  ```
  
  ### Detection Confidence
  
  ```python
  # Minimum confidence threshold (0.0 to 1.0)
  results = model(frame, conf=0.3)  # 30% confidence
  ```
  
  ## 🎨 Customization
  
  ### Add New Object Classes
  
  1. **Train Custom YOLO Model**: Train YOLOv8 on your custom dataset
  2. **Update Real Heights**: Add object heights in `REAL_HEIGHT` dictionary
  3. **Add Icons**: Update icons in `frontend.py`:
  ```python
  icons = {
      'your_class': '🎯',  # Your emoji
      # ...
  }
  ```
  
  ### Adjust Distance Zones
  
  Edit `frontend.py`:
  ```python
  if avg_distance < 5:      # Very Close (Red)
  elif avg_distance < 15:   # Close (Yellow)
  else:                      # Far (Green)
  ```
  
  ### Change UI Colors
  
  Modify gradient colors in `frontend.py`:
  ```python
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  ```
  
  ## 🐛 Troubleshooting
  
  ### Common Issues
  
  **1. Model Not Loading**
  ```bash
  Error: YOLO model not loaded
  Solution: Ensure models/best.pt exists and is a valid YOLO model
  ```
  
  **2. Backend Not Starting**
  ```bash
  Error: Port 8000 already in use
  Solution: Kill the process or use a different port:
  uvicorn main:app --port 8001
  ```
  
  **3. Video Processing Fails**
  ```bash
  Error: Could not open video file
  Solution: Check video format and codec compatibility
  Try converting to H.264 MP4 format
  ```
  
  **4. Slow Processing**
  ```bash
  Solution: 
  - Use GPU acceleration (install CUDA)
  - Reduce video resolution
  - Lower confidence threshold
  - Use smaller YOLO model (yolov8n.pt)
  ```
  
  ### Performance Optimization
  
  **For CPU:**
  ```bash
  # Use nano model for faster processing
  model = YOLO('yolov8n.pt')
  ```
  
  **For GPU:**
  ```bash
  # Install CUDA version
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
  
  ## 📊 Example Outputs
  
  ### Detection CSV Format
  ```csv
  Frame,Timestamp (s),Object,Confidence,Position,Distance (m),BBox_X1,BBox_Y1,BBox_X2,BBox_Y2
  1,0.03,car,0.95,Center,15.5,100,200,300,400
  1,0.03,person,0.87,Left,8.2,50,150,150,350
  2,0.07,truck,0.92,Right,22.3,400,180,600,380
  ```
  
  ### Detection Summary (TXT)
  ```
  YOLO Object Detection Report
  ================================
  
  Video Information:
  - Resolution: 1920x1080
  - Duration: 30 seconds
  - Total Frames: 900
  - FPS: 30
  
  Detection Summary:
  - Total Detections: 156
  - Unique Object Types: 8
  
  Object Breakdown:
  - car: 45 detections (28.8%)
  - person: 32 detections (20.5%)
  - truck: 12 detections (7.7%)
  ```
  
  ## 🤝 Contributing
  
  Contributions are welcome! Here's how you can help:
  
  1. **Fork the repository**
  2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
  3. **Commit changes**: `git commit -m 'Add AmazingFeature'`
  4. **Push to branch**: `git push origin feature/AmazingFeature`
  5. **Open a Pull Request**
  
  ### Development Guidelines
  - Follow PEP 8 style guide
  - Add comments for complex logic
  - Update documentation for new features
  - Test thoroughly before submitting
  
  
  ## 🙏 Acknowledgments
  
  - **Ultralytics** for YOLOv8
  - **FastAPI** team for the excellent web framework
  - **Streamlit** for the beautiful UI framework
  - **OpenCV** community for computer vision tools
  
  ## 📧 Contact
  
  Muhammad Furqan iftikhar - [@furqan-iftikhar1](www.linkedin.com/in/furqan-iftikhar1)
  
  Project Link: [https://github.com/mfurqaniftikahr/YOLO-Video-Object-Detection-with-Distance-Estimation](https://github.com/mfurqaniftikhar/YOLO-Video-Object-Detection-with-Distance-Estimation)
  
  ## 🌟 Star History
  
  If you find this project useful, please consider giving it a ⭐!
  
  
  **Made with ❤️ and Python**
