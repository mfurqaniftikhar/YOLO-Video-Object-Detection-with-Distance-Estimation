import streamlit as st
import requests
import time
import os
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# CONFIG
# -------------------------
API_URL = "http://localhost:8000"
ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
MAX_FILE_SIZE_MB = 500

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="YOLO Video Processor",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem 1rem;
        font-size: 1.1rem;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-box {
        padding: 1.5rem;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 10px;
        color: #155724;
        margin: 1rem 0;
        font-weight: 500;
    }
    .error-box {
        padding: 1.5rem;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 10px;
        color: #721c24;
        margin: 1rem 0;
        font-weight: 500;
    }
    .info-box {
        padding: 1.5rem;
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        border-radius: 10px;
        color: #0c5460;
        margin: 1rem 0;
        font-weight: 500;
    }
    .warning-box {
        padding: 1.5rem;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 10px;
        color: #856404;
        margin: 1rem 0;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def check_backend_health():
    """Check if backend API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_video(video_bytes, filename):
    """Upload video to backend API"""
    try:
        files = {'video': (filename, video_bytes, 'video/mp4')}
        response = requests.post(f"{API_URL}/upload", files=files, timeout=600)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('detail', 'Unknown error')
    except requests.exceptions.Timeout:
        return False, "Request timeout - video processing took too long"
    except Exception as e:
        return False, str(e)

def download_video(output_filename):
    """Download processed video from backend"""
    try:
        response = requests.get(f"{API_URL}/download/{output_filename}", timeout=60)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None

def get_metadata(job_id):
    """Get detection metadata from backend"""
    try:
        response = requests.get(f"{API_URL}/metadata/{job_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Metadata error: {str(e)}")
        return None

def validate_file(video_file):
    """Validate uploaded video file"""
    file_ext = Path(video_file.name).suffix.lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    file_size_mb = video_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f}MB). Max size: {MAX_FILE_SIZE_MB}MB"
    
    return True, "Valid"

def display_detection_card(obj_type, detections):
    """Display individual detection card with arrow and distance"""
    icons = {
        'person': '👤', 'car': '🚗', 'truck': '🚚', 'bus': '🚌',
        'motorcycle': '🏍️', 'bicycle': '🚲', 'bike': '🚲',
        'dog': '🐕', 'cat': '🐱', 'bird': '🐦', 'tree': '🌳',
        'traffic light': '🚦', 'stop sign': '🛑'
    }
    icon = icons.get(obj_type.lower(), '📦')
    
    # Get average distance
    distances = [d.get('distance') for d in detections if d.get('distance')]
    avg_distance = sum(distances) / len(distances) if distances else None
    
    # Get most common position
    positions = [d.get('position') for d in detections]
    most_common_position = max(set(positions), key=positions.count) if positions else 'Center'
    
    # Position arrow
    position_arrows = {'Left': '⬅️', 'Right': '➡️', 'Center': '⬆️'}
    arrow = position_arrows.get(most_common_position, '⬆️')
    
    # Distance color coding
    if avg_distance:
        if avg_distance < 5:
            distance_color = "🔴"
            distance_text = f"{avg_distance:.1f}m (Very Close)"
        elif avg_distance < 15:
            distance_color = "🟡"
            distance_text = f"{avg_distance:.1f}m (Close)"
        else:
            distance_color = "🟢"
            distance_text = f"{avg_distance:.1f}m (Far)"
    else:
        distance_color = "⚪"
        distance_text = "Distance Unknown"
    
    # Average confidence
    confidences = [d.get('confidence', 0) for d in detections]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Create card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 2rem;">{icon}</div>
            <div style="font-size: 1.5rem;">{arrow}</div>
        </div>
        <h3 style="margin: 0.5rem 0; color: white;">{obj_type.capitalize()}</h3>
        <div style="font-size: 1.3rem; font-weight: bold; margin: 0.5rem 0;">
            {distance_color} {distance_text}
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; opacity: 0.9;">
            <span>Position: {most_common_position}</span>
            <span>Count: {len(detections)}</span>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.8;">
            Confidence: {avg_confidence:.0%}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_live_detections(metadata):
    """Display live detection information with arrows and distances"""
    if not metadata or 'detections' not in metadata:
        return
    
    st.markdown("### 🎯 Live Detection Feed")
    
    # Get latest detections
    latest_detections = []
    if metadata.get('detections'):
        for frame_data in metadata['detections'][:5]:
            for det in frame_data.get('detections', []):
                latest_detections.append(det)
    
    if not latest_detections:
        st.info("No detections found in preview frames")
        return
    
    # Group by object type
    grouped_detections = {}
    for det in latest_detections:
        obj_type = det.get('class', 'Unknown')
        if obj_type not in grouped_detections:
            grouped_detections[obj_type] = []
        grouped_detections[obj_type].append(det)
    
    # Display in columns
    col1, col2 = st.columns(2)
    items = list(grouped_detections.items())
    mid_point = len(items) // 2 if len(items) > 1 else 1
    
    with col1:
        for obj_type, detections in items[:mid_point]:
            display_detection_card(obj_type, detections)
    
    with col2:
        for obj_type, detections in items[mid_point:]:
            display_detection_card(obj_type, detections)

def save_detections_to_csv(metadata, job_id):
    """Save detection data to CSV file"""
    try:
        if not metadata or 'detections' not in metadata:
            return None
        
        rows = []
        for frame_data in metadata['detections']:
            frame_num = frame_data.get('frame', 0)
            timestamp = frame_data.get('timestamp', 0)
            
            for detection in frame_data.get('detections', []):
                row = {
                    'Frame': frame_num,
                    'Timestamp (s)': timestamp,
                    'Object': detection.get('class', 'Unknown'),
                    'Confidence': detection.get('confidence', 0),
                    'Position': detection.get('position', 'Unknown'),
                    'Distance (m)': detection.get('distance', 'N/A'),
                    'BBox_X1': detection.get('bbox', [0, 0, 0, 0])[0],
                    'BBox_Y1': detection.get('bbox', [0, 0, 0, 0])[1],
                    'BBox_X2': detection.get('bbox', [0, 0, 0, 0])[2],
                    'BBox_Y2': detection.get('bbox', [0, 0, 0, 0])[3]
                }
                rows.append(row)
        
        if not rows:
            return None
        
        df = pd.DataFrame(rows)
        csv_filename = f"detections_{job_id}.csv"
        csv_data = df.to_csv(index=False)
        
        return csv_data, csv_filename, df
    except Exception as e:
        st.error(f"Error saving CSV: {e}")
        return None

def display_detection_summary(metadata):
    """Display detection summary with visualizations"""
    if not metadata or 'detection_summary' not in metadata:
        return
    
    detection_summary = metadata['detection_summary']
    
    if not detection_summary:
        st.warning("No objects detected in this video")
        return
    
    st.markdown("### 📊 Detection Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 0.9rem; opacity: 0.9;">Total Detections</div>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">{metadata.get('total_detections', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 0.9rem; opacity: 0.9;">Unique Objects</div>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">{metadata.get('unique_classes', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        video_duration = metadata.get('video_info', {}).get('duration_seconds', 0)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 0.9rem; opacity: 0.9;">Video Duration</div>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">{video_duration}s</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_frames = metadata.get('video_info', {}).get('total_frames', 0)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 0.9rem; opacity: 0.9;">Total Frames</div>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">{total_frames}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Save detections buttons
    st.markdown("### 💾 Save Detection Data")
    
    save_col1, save_col2, save_col3 = st.columns([1, 1, 1])
    
    with save_col1:
        csv_result = save_detections_to_csv(metadata, metadata.get('job_id', 'unknown'))
        if csv_result:
            csv_data, csv_filename, df = csv_result
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=True
            )
    
    with save_col2:
        json_data = json.dumps(metadata, indent=2)
        st.download_button(
            label="📥 Download JSON Data",
            data=json_data,
            file_name=f"metadata_{metadata.get('job_id', 'unknown')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with save_col3:
        summary_text = f"""YOLO Object Detection Report
================================

Video Information:
- Resolution: {metadata.get('video_info', {}).get('width', 0)}x{metadata.get('video_info', {}).get('height', 0)}
- Duration: {metadata.get('video_info', {}).get('duration_seconds', 0)} seconds
- Total Frames: {metadata.get('video_info', {}).get('total_frames', 0)}

Detection Summary:
- Total Detections: {metadata.get('total_detections', 0)}
- Unique Object Types: {metadata.get('unique_classes', 0)}

Object Breakdown:
"""
        for obj_name, count in sorted(detection_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / metadata.get('total_detections', 1) * 100)
            summary_text += f"- {obj_name}: {count} ({percentage:.1f}%)\n"
        
        st.download_button(
            label="📥 Download Summary (TXT)",
            data=summary_text,
            file_name=f"summary_{metadata.get('job_id', 'unknown')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("#### 📈 Detection Count")
        df_summary = pd.DataFrame(
            list(detection_summary.items()),
            columns=['Object', 'Count']
        ).sort_values('Count', ascending=False)
        
        fig_bar = px.bar(df_summary, x='Object', y='Count', color='Count',
                         color_continuous_scale='Viridis', text='Count')
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with viz_col2:
        st.markdown("#### 🥧 Distribution")
        fig_pie = px.pie(df_summary, values='Count', names='Object')
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------
# MAIN APP
# -------------------------
def main():
    st.title("🎥 YOLO Video Object Detection System")
    st.markdown("### Upload and analyze videos with AI-powered object detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        if check_backend_health():
            st.markdown('<div class="success-box">✅ Backend: Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ Backend: Offline</div>', unsafe_allow_html=True)
            st.error("Please start the FastAPI server")
            st.code("python main.py")
            st.stop()
        
        st.markdown("---")
        st.header("🎯 Detectable Objects")
        st.markdown("""
        - 👤 Person
        - 🚗 Vehicles (Car, Truck, Bus)
        - 🏍️ Bikes (Motorcycle, Bicycle)
        - 🐕 Animals (Dog, Cat, Bird)
        - 🌳 Environment
        """)
    
    # Main tabs
    tab1, tab2 = st.tabs(["📤 Upload & Process", "📊 Results & Analysis"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📤 Upload Video")
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']
            )
            
            if uploaded_file is not None:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.success(f"✅ File: **{uploaded_file.name}**")
                st.info(f"📊 Size: **{file_size_mb:.2f} MB**")
                
                is_valid, message = validate_file(uploaded_file)
                if not is_valid:
                    st.error(f"❌ {message}")
                    st.stop()
                
                video_bytes = uploaded_file.read()
                st.session_state['video_bytes'] = video_bytes
                st.session_state['video_name'] = uploaded_file.name
                
                st.subheader("📹 Original Video")
                try:
                    st.video(video_bytes)
                except:
                    st.warning("Cannot preview video")
                
                st.markdown("---")
                if st.button("🚀 Process Video", use_container_width=True):
                    with st.spinner("Processing..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("⏳ Uploading...")
                        progress_bar.progress(10)
                        
                        success, result = upload_video(video_bytes, uploaded_file.name)
                        
                        if success:
                            progress_bar.progress(100)
                            status_text.text("✅ Complete!")
                            
                            st.session_state['processed_result'] = result
                            st.session_state['processing_complete'] = True
                            
                            if 'job_id' in result:
                                metadata = get_metadata(result['job_id'])
                                if metadata:
                                    st.session_state['metadata'] = metadata
                            
                            st.success("✅ Video processed successfully!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result}")
            else:
                st.info("👆 Upload a video to get started")
        
        with col2:
            st.header("📥 Download Results")
            
            if 'processing_complete' in st.session_state and st.session_state['processing_complete']:
                result = st.session_state['processed_result']
                st.markdown('<div class="success-box">✅ Processing Complete!</div>', unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Total Detections", result.get('total_detections', 0))
                with col_b:
                    st.metric("Object Types", len(result.get('detection_summary', {})))
                
                output_filename = result.get('output_filename')
                if output_filename:
                    st.markdown("---")
                    with st.spinner("Loading video..."):
                        video_content = download_video(output_filename)
                    
                    if video_content:
                        st.video(video_content)
                        st.download_button(
                            label="💾 Download Processed Video",
                            data=video_content,
                            file_name=output_filename,
                            mime="video/mp4",
                            use_container_width=True
                        )
                
                st.markdown("---")
                if st.button("🔄 Process Another Video", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            else:
                st.info("👈 Upload and process a video first")
    
    with tab2:
        st.header("📊 Detection Analysis")
        
        if 'metadata' in st.session_state:
            metadata = st.session_state['metadata']
            display_live_detections(metadata)
            st.markdown("---")
            display_detection_summary(metadata)
        else:
            st.warning("⚠️ No detection data. Process a video first.")

if __name__ == "__main__":
    main()