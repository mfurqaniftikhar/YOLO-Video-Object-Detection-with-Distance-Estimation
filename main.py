import os
import cv2
import time
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# -------------------------
#  FOLDER CONFIG
# -------------------------
UPLOAD_FOLDER = "uploads"
JOBS_FOLDER = "jobs"
MODEL_PATH = "models/best.pt"
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JOBS_FOLDER, exist_ok=True)

# Thread pool for video processing
executor = ThreadPoolExecutor(max_workers=2)

# -------------------------
#  LOAD YOLO MODEL
# -------------------------
print("Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# -------------------------
# SIMPLE DISTANCE ESTIMATION
# -------------------------
REAL_HEIGHT = {
    "person": 1.7,
    "car": 1.4,
    "truck": 3.0,
    "bike": 1.2,
    "bicycle": 1.2,
    "motorcycle": 1.2,
    "bus": 3.2,
    "animal": 0.8,
    "dog": 0.6,
    "cat": 0.3,
    "bird": 0.2,
    "tree": 4.0
}
FOCAL_LENGTH = 700

def estimate_distance(cls, pixel_height):
    """Estimate distance based on object class and pixel height"""
    if pixel_height <= 0:
        return None
    
    cls_lower = cls.lower()
    if cls_lower not in REAL_HEIGHT:
        return None
    
    distance = (REAL_HEIGHT[cls_lower] * FOCAL_LENGTH) / pixel_height
    return round(distance, 2)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def get_position_text(x_center, frame_width):
    """Determine object position (left, center, right)"""
    left_threshold = frame_width * 0.33
    right_threshold = frame_width * 0.66
    
    if x_center < left_threshold:
        return "Left"
    elif x_center > right_threshold:
        return "Right"
    else:
        return "Center"

def draw_info_panel(frame, detections_info):
    """Draw information panel at top of frame"""
    if not detections_info:
        return
    
    height, width = frame.shape[:2]
    
    # Calculate panel height based on number of detections (max 12 items to show)
    visible_detections = detections_info[:12]
    panel_height = 60 + (len(visible_detections) * 35)
    
    # Limit panel height to not cover too much of the frame
    panel_height = min(panel_height, height // 2)
    
    # Create semi-transparent dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw top border
    cv2.line(frame, (0, panel_height-2), (width, panel_height-2), (0, 255, 255), 3)
    
    # Draw title with icon
    title = "DETECTED OBJECTS"
    cv2.putText(frame, title, (15, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
    
    # Draw count badge
    count_text = f"Total: {len(detections_info)}"
    (count_w, count_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    badge_x = width - count_w - 30
    cv2.rectangle(frame, (badge_x - 10, 15), (width - 15, 45), (0, 255, 255), -1)
    cv2.putText(frame, count_text, (badge_x, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Draw detection info with better formatting
    y_offset = 75
    for i, info in enumerate(visible_detections):
        # Alternate background colors for better readability
        if i % 2 == 0:
            cv2.rectangle(frame, (5, y_offset - 25), (width - 5, y_offset + 5), (30, 30, 30), -1)
        
        # Draw bullet point
        cv2.circle(frame, (20, y_offset - 10), 5, (0, 255, 255), -1)
        
        # Draw text with shadow for better visibility
        cv2.putText(frame, info, (40, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(frame, info, (40, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        y_offset += 35
    
    # If more detections exist, show indicator
    if len(detections_info) > 12:
        more_text = f"... and {len(detections_info) - 12} more"
        cv2.putText(frame, more_text, (40, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

# -------------------------
# VIDEO PROCESS FUNCTION
# -------------------------
def process_video(input_path, output_path, job_id):
    """Process video with YOLO detection and distance estimation"""
    cap = None
    out = None
    
    # Create metadata file path
    metadata_path = os.path.join(JOBS_FOLDER, f"{job_id}_metadata.json")
    
    try:
        print(f"Opening video file: {input_path}")
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info - Width: {width}, Height: {height}, FPS: {fps}, Frames: {total_frames}")
        
        # Handle invalid FPS
        if fps == 0 or fps is None or fps > 120 or fps < 1:
            print("Invalid FPS detected, using default 30.0")
            fps = 30.0
        
        # Try multiple codecs for better compatibility
        fourcc_options = [
            ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # H.264
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # XVID
        ]
        
        out = None
        for codec_name, fourcc in fourcc_options:
            print(f"Trying codec: {codec_name}")
            temp_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if temp_out.isOpened():
                out = temp_out
                print(f"Successfully initialized with codec: {codec_name}")
                break
            else:
                temp_out.release()
        
        if out is None or not out.isOpened():
            raise Exception(f"Could not create output video with any codec")
        
        frame_count = 0
        processed_count = 0
        
        # Store all detections for metadata
        all_detections = []
        detection_summary = {}
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No more frames to read")
                break
            
            frame_count += 1
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            try:
                # Store detection information for top corner display
                detections_info = []
                frame_detections = []
                
                # YOLO inference with error handling
                results = model(frame, verbose=False, conf=0.3)
                
                for r in results:
                    boxes = r.boxes
                    
                    if boxes is None or len(boxes) == 0:
                        continue
                    
                    for box in boxes:
                        try:
                            # Get class and coordinates
                            cls_id = int(box.cls.cpu().numpy()[0])
                            cls_name = model.names[cls_id]
                            
                            # Get bounding box
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Ensure coordinates are within frame bounds
                            x1 = max(0, min(x1, width - 1))
                            y1 = max(0, min(y1, height - 1))
                            x2 = max(0, min(x2, width - 1))
                            y2 = max(0, min(y2, height - 1))
                            
                            # Calculate center and height
                            x_center = (x1 + x2) / 2
                            h = y2 - y1
                            
                            if h <= 0:
                                continue
                            
                            # Get confidence
                            conf = float(box.conf.cpu().numpy()[0])
                            
                            # Estimate distance
                            distance = estimate_distance(cls_name, h)
                            
                            # Determine position
                            position = get_position_text(x_center, width)
                            
                            # Create detection data
                            detection_data = {
                                "class": cls_name,
                                "confidence": round(conf, 2),
                                "position": position,
                                "distance": distance,
                                "bbox": [x1, y1, x2, y2]
                            }
                            
                            frame_detections.append(detection_data)
                            
                            # Update summary count
                            if cls_name not in detection_summary:
                                detection_summary[cls_name] = 0
                            detection_summary[cls_name] += 1
                            
                            # Create description for top panel
                            if distance:
                                info_text = f"{position}: {cls_name} at {distance}m (conf: {conf:.2f})"
                                detections_info.append(info_text)
                            else:
                                info_text = f"{position}: {cls_name} (conf: {conf:.2f})"
                                detections_info.append(info_text)
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Create label for bounding box
                            label = f"{cls_name} {conf:.2f}"
                            if distance:
                                label += f" {distance}m"
                            
                            # Draw label background
                            (label_w, label_h), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                            )
                            
                            # Ensure label stays within frame
                            label_y = max(y1 - 10, label_h + 5)
                            
                            cv2.rectangle(
                                frame,
                                (x1, label_y - label_h - 5),
                                (x1 + label_w, label_y),
                                (0, 255, 0),
                                -1
                            )
                            
                            # Draw text
                            cv2.putText(
                                frame, label, (x1, label_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                            )
                        except Exception as box_error:
                            print(f"Error processing box: {box_error}")
                            continue
                
                # Store frame detections
                if frame_detections:
                    all_detections.append({
                        "frame": frame_count,
                        "timestamp": round(frame_count / fps, 2),
                        "detections": frame_detections
                    })
                
                # Draw info panel at top if there are detections
                if detections_info:
                    draw_info_panel(frame, detections_info)
                
                # Write frame
                out.write(frame)
                processed_count += 1
                
            except Exception as frame_error:
                print(f"Error processing frame {frame_count}: {frame_error}")
                # Write original frame without annotations
                out.write(frame)
                continue
        
        print(f"Video processing complete: {processed_count}/{frame_count} frames processed successfully")
        
        # Save metadata
        metadata = {
            "job_id": job_id,
            "input_file": os.path.basename(input_path),
            "output_file": os.path.basename(output_path),
            "video_info": {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": frame_count,
                "duration_seconds": round(frame_count / fps, 2)
            },
            "detection_summary": detection_summary,
            "total_detections": sum(detection_summary.values()),
            "unique_classes": len(detection_summary),
            "detections": all_detections[:100]  # Store first 100 frames for preview
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
        
        return True, metadata
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        raise e
        
    finally:
        if cap is not None:
            cap.release()
            print("Released video capture")
        if out is not None:
            out.release()
            print("Released video writer")

# -------------------------
# FASTAPI APP
# -------------------------
app = FastAPI(
    title="YOLO Video Processing API",
    description="Upload videos for object detection and distance estimation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "YOLO Video Processing API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload": "Upload and process video",
            "GET /download/{filename}": "Download processed video",
            "GET /metadata/{job_id}": "Get detection metadata",
            "GET /health": "Health check",
            "DELETE /cleanup": "Clean up old files"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if model else None,
        "upload_folder": os.path.exists(UPLOAD_FOLDER),
        "jobs_folder": os.path.exists(JOBS_FOLDER)
    }

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    """Upload and process video file"""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    
    # Validate file extension
    file_ext = Path(video.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Create unique job ID
    job_id = f"{int(time.time())}_{os.urandom(4).hex()}"
    
    # Create unique filename
    safe_filename = "".join(c for c in video.filename if c.isalnum() or c in "._- ")
    filename = f"{job_id}_{safe_filename}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_filename = f"detected_{filename}"
    output_path = os.path.join(JOBS_FOLDER, output_filename)
    
    print(f"Processing upload: {video.filename}")
    print(f"Job ID: {job_id}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    
    try:
        # Save uploaded file
        print("Saving uploaded file...")
        with open(input_path, "wb") as f:
            content = await video.read()
            
            # Check file size
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024)}MB"
                )
            
            f.write(content)
        
        print(f"File uploaded successfully: {input_path} ({len(content)} bytes)")
        
        # Verify file exists and is readable
        if not os.path.exists(input_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        # Process video in thread pool to avoid blocking
        print("Starting video processing...")
        loop = asyncio.get_event_loop()
        result, metadata = await loop.run_in_executor(
            executor, process_video, input_path, output_path, job_id
        )
        
        # Check if output file was created
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Video processing failed - output file not created")
        
        output_size = os.path.getsize(output_path)
        print(f"Processing complete! Output file: {output_path} ({output_size} bytes)")
        
        return JSONResponse({
            "message": "Video processed successfully",
            "job_id": job_id,
            "input_file": filename,
            "output_filename": output_filename,
            "download_url": f"/download/{output_filename}",
            "metadata_url": f"/metadata/{job_id}",
            "input_size_mb": round(len(content) / (1024*1024), 2),
            "output_size_mb": round(output_size / (1024*1024), 2),
            "detection_summary": metadata.get("detection_summary", {}),
            "total_detections": metadata.get("total_detections", 0)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in upload endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up files on error
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
                print(f"Cleaned up input file: {input_path}")
            except:
                pass
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Cleaned up output file: {output_path}")
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download processed video"""
    
    # Security: prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(JOBS_FOLDER, safe_filename)
    
    print(f"Download requested: {safe_filename}")
    print(f"File path: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    print(f"Serving file: {file_path}")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=safe_filename,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"'
        }
    )

@app.get("/metadata/{job_id}")
async def get_metadata(job_id: str):
    """Get detection metadata for a processed video"""
    
    # Security: prevent path traversal
    safe_job_id = "".join(c for c in job_id if c.isalnum() or c in "_-")
    metadata_path = os.path.join(JOBS_FOLDER, f"{safe_job_id}_metadata.json")
    
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Metadata not found")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metadata: {str(e)}")

@app.delete("/cleanup")
async def cleanup_files():
    """Clean up old files (optional maintenance endpoint)"""
    try:
        
        current_time = time.time()
        deleted_count = 0
        
        for folder in [UPLOAD_FOLDER, JOBS_FOLDER]:
            if not os.path.exists(folder):
                continue
                
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > 3600:  # 1 hour
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                            print(f"Deleted old file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting {file_path}: {e}")
        
        return {"message": f"Cleaned up {deleted_count} old files"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    print("="*50)
    print("YOLO Video Processing API Starting...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Jobs folder: {JOBS_FOLDER}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print("="*50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )