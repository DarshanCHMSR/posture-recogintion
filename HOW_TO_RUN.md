# 🤖 AI Posture Monitor - How to Run Guide

## 📋 Quick Start Guide

### 1. **Prerequisites**
- ✅ Python 3.8+ (You have Python 3.11.9)  
- 📷 Webcam or video files for testing
- 💡 Well-lit indoor environment for best results

### 2. **Setup (Automated)**
Run the setup script to install everything automatically:

```bash
# Navigate to project directory
cd "c:\Users\Darsh\OneDrive\Desktop\full stack\Internship-demo\elderly\AI-Posture-Monitor"

# Activate virtual environment
.\myevn\Scripts\Activate.ps1

# Run automated setup
python setup_and_run.py
```

### 3. **Setup (Manual)**
If you prefer manual setup:

```bash
# Activate virtual environment
.\myevn\Scripts\Activate.ps1

# Install dependencies
pip install numpy opencv-python mediapipe pandas scikit-learn matplotlib

# Install AI Posture Monitor package
cd ai_posture_monitor_package
pip install .
cd ..
```

### 4. **Test Installation**
```python
python -c "import ai_posture_monitor as pm; print('✅ Success! Version:', pm.__version__)"
```

---

## 🚀 Running the Project

### **Option 1: Webcam Testing (Recommended for First Run)**
```bash
python test_webcam.py
```
- Press **'q'** to quit
- Shows real-time posture detection
- Works without video files

### **Option 2: Video File Analysis**

#### Posture Detection Only:
```bash
python ai_posture_monitor_package/tests/predict_video_static_pose.py VIDEO_FILE 1 0.8
```

#### Fall Detection:
```bash
python ai_posture_monitor_package/tests/predict_falls.py VIDEO_FILE 1 0.8
```

**Parameters:**
- `VIDEO_FILE`: Path to your video file
- `1`: Enable predictions (use `0` to disable)
- `0.8`: Scaling factor (0.5-1.0, lower = faster)

### **Option 3: Using in Python Code**

```python
import ai_posture_monitor as pm

# Initialize pose estimation
pe = pm.PoseEstimation()

# For posture detection only
pe.process_video(
    video_file="your_video.mp4",  # or None for webcam
    plot_results=True,
    predict_fall=False
)

# For fall detection
pe.process_video(
    video_file="your_video.mp4",  # or None for webcam  
    plot_results=True,
    predict_fall=True
)
```

---

## 🎯 Key Features

- **🏃 Real-time Activity Monitoring**: Tracks movements continuously
- **🧍 Pose Detection**: Identifies standing, sitting, lying postures  
- **🚨 Fall Detection**: Detects falls with minimal false alarms
- **🧠 Fuzzy Logic Analysis**: Uses advanced logic for accurate interpretation
- **👥 Elderly Care Focus**: Designed specifically for elderly monitoring
- **🏠 Indoor Optimized**: Works best in well-lit indoor environments

---

## 📁 Project Structure

```
AI-Posture-Monitor/
├── 📦 ai_posture_monitor_package/     # Main package
├── 🧪 tests/                          # Test scripts  
├── 🏷️ labels/                         # Sample labels
├── 🔬 proof-of-concept/               # Research scripts
├── 🐍 myevn/                          # Virtual environment
├── 📋 Requirements.txt                # Dependencies
├── 🚀 setup_and_run.py               # Automated setup
└── 🧪 test_webcam.py                 # Quick webcam test
```

---

## 🛠️ Advanced Usage

### **Creating New Dataset**
1. **Extract keyframes from video:**
   ```bash
   python proof-of-concept/pose-estimation-on-video/groundtruth.py ./your_video.mp4
   ```

2. **Create label CSV file** with format:
   ```csv
   start_time,end_time,action,is_fall
   0,10,Stand,False
   11,32,Stand,False
   33,35,Stand-Lie,True
   36,40,Lie,True
   ```

3. **Validate labels:**
   ```bash
   python proof-of-concept/pose-estimation-on-video/analyze_manual_label.py ./labels/your_labels.csv 0
   ```

4. **Run fall detection:**
   ```bash
   python proof-of-concept/pose-estimation-on-video/predict_fall.py ./your_video.mp4 1 1 ./labels/your_labels.csv
   ```

### **Utility Scripts**
- **Plot histogram:** `python proof-of-concept/pose-estimation-on-video/plot_his.py VIDEO_FILE 1`
- **Frame differencing:** `python proof-of-concept/pose-estimation-on-video/frame_diff.py VIDEO_FILE 1`
- **Visualize results:** `python proof-of-concept/pose-estimation-on-video/fall_plot.py RESULTS.csv`

---

## ⚡ Performance Tips

- **📹 Camera Quality**: Use HD camera for better pose detection
- **💡 Lighting**: Ensure good lighting (avoid shadows/backlighting)  
- **🏠 Environment**: Indoor environments work best
- **⚖️ Scaling Factor**: 
  - `0.5-0.7`: Faster processing, good for real-time
  - `0.8-1.0`: Higher accuracy, slower processing
- **🖥️ Hardware**: GPU acceleration available with proper setup

---

## 🆘 Troubleshooting

### **Common Issues:**

1. **❌ ImportError: No module named 'ai_posture_monitor'**
   ```bash
   cd ai_posture_monitor_package
   pip install . --force-reinstall
   ```

2. **❌ Camera not detected**
   - Check camera permissions
   - Try different camera index: `video_file=1` instead of `None`

3. **❌ Poor detection accuracy**
   - Improve lighting conditions
   - Increase scaling factor to 0.8-1.0
   - Ensure subject is fully visible in frame

4. **❌ Slow performance**
   - Reduce scaling factor to 0.5-0.7
   - Close other applications using camera
   - Use lower resolution video

### **Get Help:**
- Check the original README.md for detailed documentation
- Review proof-of-concept scripts for examples
- Ensure all dependencies are properly installed

---

## 📊 Expected Output

When running successfully, you'll see:
- **Real-time video window** with pose landmarks
- **Posture predictions** (Stand/Sit/Lie)
- **Fall detection alerts** (if enabled)
- **CSV output files** with detailed results
- **Visualization plots** (if plot_results=True)

---

**🎉 You're all set! Start with `python test_webcam.py` for a quick test.**
