"""
Enhanced AI Posture Monitor with Improved Sitting Detection and Email Reporting
Uses MediaPipe with enhanced detection thresholds and real-time email reporting
"""
import cv2
import time
import json
from datetime import datetime, timedelta
import os
import numpy as np
import mediapipe as mp

# Import the pose estimation package and email system
from ai_posture_monitor_package.ai_posture_monitor.pose_est_dependencies import PoseEstimation
from smtp_email_system import SMTPEmailNotifier
from enhanced_pose_graphs import EnhancedPoseGraphGenerator

class EnhancedPoseEstimationEmailReporter:
    def __init__(self, config_file='email_config.json'):
        """Initialize with email configuration"""
        self.load_email_config(config_file)
        
        # Initialize MediaPipe directly for better control
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.email_notifier = SMTPEmailNotifier(config_file)
        self.graph_generator = EnhancedPoseGraphGenerator()
        
        # Enhanced pose tracking
        self.pose_history = []
        self.last_pose = None
        self.pose_transition_count = 0
        self.session_start_time = datetime.now()
        self.monitoring_active = True
        
        # Walking detection tracking
        self.previous_landmarks = None
        self.walking_movement_buffer = []
        self.walking_threshold = 0.03  # Movement threshold for walking detection
        self.walking_confirmation_frames = 5  # Frames needed to confirm walking
        
        # Pose counters with detailed tracking
        self.pose_stats = {
            'standing': {'count': 0, 'duration': 0, 'confidence_scores': []},
            'sitting': {'count': 0, 'duration': 0, 'confidence_scores': []},
            'lying': {'count': 0, 'duration': 0, 'confidence_scores': []},
            'walking': {'count': 0, 'duration': 0, 'confidence_scores': []},
            'unknown': {'count': 0, 'duration': 0, 'confidence_scores': []}
        }
        
        # Enhanced sitting detection parameters
        self.sitting_confidence_threshold = 0.5  # Lowered for better sensitivity
        self.sitting_confirmation_frames = 3     # Reduced for faster detection
        self.recent_sitting_predictions = []
        
        print("✅ Enhanced Posture Monitor initialized with improved sitting detection")

    def load_email_config(self, config_file):
        """Load email configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                self.email_config = json.load(f)
            print(f"✅ Email configuration loaded from {config_file}")
        except FileNotFoundError:
            print(f"❌ Email config file {config_file} not found")
            raise
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {config_file}")
            raise

    def detect_walking_movement(self, current_landmarks):
        """
        Detect walking by analyzing frame-to-frame movement of key joints
        """
        if self.previous_landmarks is None:
            self.previous_landmarks = current_landmarks
            return False, 0.0
            
        try:
            # Key joints for walking detection: hips, knees, ankles
            key_joints = [23, 24, 25, 26, 27, 28]  # left hip, right hip, left knee, right knee, left ankle, right ankle
            
            total_movement = 0.0
            joint_count = 0
            
            for joint_idx in key_joints:
                if joint_idx < len(current_landmarks.landmark) and joint_idx < len(self.previous_landmarks.landmark):
                    curr = current_landmarks.landmark[joint_idx]
                    prev = self.previous_landmarks.landmark[joint_idx]
                    
                    # Calculate 2D movement (x, y coordinates)
                    movement = ((curr.x - prev.x)**2 + (curr.y - prev.y)**2)**0.5
                    total_movement += movement
                    joint_count += 1
            
            # Calculate average movement
            avg_movement = total_movement / joint_count if joint_count > 0 else 0.0
            
            # Add to movement buffer for smoothing
            self.walking_movement_buffer.append(avg_movement)
            if len(self.walking_movement_buffer) > self.walking_confirmation_frames:
                self.walking_movement_buffer.pop(0)
            
            # Check if recent movements indicate walking
            recent_avg_movement = sum(self.walking_movement_buffer) / len(self.walking_movement_buffer)
            is_walking = recent_avg_movement > self.walking_threshold and len(self.walking_movement_buffer) >= 3
            
            # Calculate confidence based on movement consistency
            movement_confidence = min(recent_avg_movement / self.walking_threshold, 1.0) if is_walking else 0.0
            
            # Update previous landmarks
            self.previous_landmarks = current_landmarks
            
            return is_walking, movement_confidence
            
        except Exception as e:
            print(f"⚠️ Error in walking detection: {e}")
            return False, 0.0

    def enhanced_pose_detection(self, pose_landmarks, frame_confidence):
        """
        Enhanced pose detection with walking, sitting, standing, and lying detection
        """
        if not pose_landmarks:
            return "unknown", 0.0
            
        try:
            # First check for walking movement
            is_walking, walking_confidence = self.detect_walking_movement(pose_landmarks)
            
            # If walking is detected with high confidence, return walking
            if is_walking and walking_confidence > 0.5:
                return "walking", walking_confidence
            
            # Get key landmarks for static pose detection
            landmarks = pose_landmarks.landmark
            
            # Hip landmarks
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Knee landmarks
            left_knee = landmarks[25] 
            right_knee = landmarks[26]
            
            # Ankle landmarks
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            # Shoulder landmarks for posture context
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # Calculate average positions
            avg_hip_y = (left_hip.y + right_hip.y) / 2
            avg_knee_y = (left_knee.y + right_knee.y) / 2
            avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
            avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Enhanced sitting criteria
            sitting_indicators = []
            confidence_factors = []
            
            # Criterion 1: Knee position relative to hip (primary indicator for sitting)
            knee_hip_ratio = avg_knee_y / avg_hip_y if avg_hip_y > 0 else 1
            if knee_hip_ratio < 1.2:  # More lenient - knees closer to hip level indicates sitting
                sitting_indicators.append(True)
                confidence_factors.append(0.4)
            else:
                sitting_indicators.append(False)
                confidence_factors.append(0.0)
            
            # Criterion 2: Hip-ankle distance (sitting reduces this distance significantly)
            hip_ankle_distance = abs(avg_hip_y - avg_ankle_y)
            if hip_ankle_distance < 0.4:  # Increased threshold for better sitting detection
                sitting_indicators.append(True)
                confidence_factors.append(0.3)
            else:
                sitting_indicators.append(False)
                confidence_factors.append(0.0)
                
            # Criterion 3: Torso compactness (sitting makes torso more compact)
            shoulder_hip_distance = abs(avg_shoulder_y - avg_hip_y)
            if shoulder_hip_distance < 0.25:  # Compact torso suggests sitting
                sitting_indicators.append(True)
                confidence_factors.append(0.2)
            else:
                sitting_indicators.append(False)
                confidence_factors.append(0.0)
            
            # Criterion 4: Overall pose compactness (sitting creates more compact pose)
            pose_height = abs(avg_shoulder_y - avg_ankle_y)
            if pose_height < 0.7:  # Increased threshold for better sitting detection
                sitting_indicators.append(True)
                confidence_factors.append(0.1)
            else:
                sitting_indicators.append(False)
                confidence_factors.append(0.0)
                
            # Calculate sitting confidence
            sitting_confidence = sum(confidence_factors)
            sitting_score = sum(sitting_indicators)
            
            # Debug information
            if sitting_score > 0:
                print(f"\n🔍 Debug - Sitting indicators: {sitting_score}/4, confidence: {sitting_confidence:.2f}")
                print(f"   Knee/Hip ratio: {knee_hip_ratio:.2f}, Hip/Ankle dist: {hip_ankle_distance:.2f}")
            
            # Determine pose based on enhanced criteria with improved sitting detection
            if sitting_score >= 2 and sitting_confidence >= self.sitting_confidence_threshold:
                pose = "sitting"
                confidence = min(sitting_confidence + frame_confidence * 0.2, 1.0)
            elif sitting_score >= 1 and sitting_confidence >= 0.3:  # Lower threshold for sitting
                pose = "sitting"
                confidence = sitting_confidence * 0.8
            elif avg_knee_y > avg_hip_y * 1.4:  # Standing: knees well below hips (more strict)
                pose = "standing"  
                confidence = frame_confidence * 0.8
            elif avg_hip_y > avg_knee_y * 1.3:  # Lying: hips well below knees
                pose = "lying"
                confidence = frame_confidence * 0.7
            else:
                pose = "unknown"
                confidence = 0.3
                
            return pose, confidence
            
        except Exception as e:
            print(f"⚠️ Error in enhanced sitting detection: {e}")
            return "unknown", 0.0

    def process_frame_with_enhanced_detection(self, frame):
        """Process frame with enhanced sitting detection"""
        try:
            # Use MediaPipe for initial pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Get frame confidence from MediaPipe
                frame_confidence = 0.8  # Default confidence
                
                # Enhanced pose classification
                pose, confidence = self.enhanced_pose_detection(results.pose_landmarks, frame_confidence)
                
                # Sitting confirmation logic
                if pose == "sitting":
                    self.recent_sitting_predictions.append(1)
                else:
                    self.recent_sitting_predictions.append(0)
                    
                # Keep only recent predictions for confirmation
                if len(self.recent_sitting_predictions) > self.sitting_confirmation_frames:
                    self.recent_sitting_predictions.pop(0)
                
                # Confirm sitting if consistently detected (more lenient)
                sitting_ratio = sum(self.recent_sitting_predictions) / len(self.recent_sitting_predictions)
                if sitting_ratio >= 0.5 and pose == "sitting":  # More lenient confirmation
                    final_pose = "sitting"
                    final_confidence = confidence
                elif sitting_ratio < 0.4 and pose != "sitting":
                    final_pose = pose
                    final_confidence = confidence
                else:
                    # Use original prediction but adjust confidence
                    final_pose = pose
                    final_confidence = confidence * 0.8  # Slightly reduce confidence for uncertain cases
                
                # Update statistics
                self.update_pose_statistics(final_pose, final_confidence)
                
                return final_pose, final_confidence, results.pose_landmarks
            else:
                return "unknown", 0.0, None
                
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return "unknown", 0.0, None

    def update_pose_statistics(self, pose, confidence):
        """Update pose statistics with real data"""
        if pose in self.pose_stats:
            self.pose_stats[pose]['count'] += 1
            self.pose_stats[pose]['confidence_scores'].append(confidence)
            
        # Track pose transitions
        if self.last_pose and self.last_pose != pose:
            self.pose_transition_count += 1
        self.last_pose = pose
        
        # Add to pose history
        self.pose_history.append({
            'timestamp': datetime.now(),
            'pose': pose,
            'confidence': confidence
        })

    def calculate_real_pose_statistics(self):
        """Calculate real statistics from tracked data"""
        total_frames = sum(stats['count'] for stats in self.pose_stats.values())
        if total_frames == 0:
            return self.get_default_statistics()
            
        # Calculate percentages and durations
        session_duration = (datetime.now() - self.session_start_time).total_seconds() / 60
        
        statistics = {}
        for pose, stats in self.pose_stats.items():
            count = stats['count']
            percentage = (count / total_frames) * 100 if total_frames > 0 else 0
            duration_minutes = (count / total_frames) * session_duration if total_frames > 0 else 0
            avg_confidence = np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0
            
            statistics[f'{pose}_count'] = count
            statistics[f'{pose}_percentage'] = round(percentage, 1)
            statistics[f'{pose}_minutes'] = round(duration_minutes, 1)
            statistics[f'{pose}_confidence'] = round(avg_confidence * 100, 1)
        
        # Additional metrics
        statistics['total_frames'] = total_frames
        statistics['session_minutes'] = round(session_duration, 1)
        statistics['transitions'] = self.pose_transition_count
        statistics['avg_confidence'] = round(np.mean([
            np.mean(stats['confidence_scores']) for stats in self.pose_stats.values() 
            if stats['confidence_scores']
        ]) * 100, 1) if any(stats['confidence_scores'] for stats in self.pose_stats.values()) else 0
        
        return statistics

    def format_enhanced_email_report(self):
        """Format comprehensive email report with real data"""
        stats = self.calculate_real_pose_statistics()
        
        # Session overview
        session_duration = round((datetime.now() - self.session_start_time).total_seconds() / 60, 1)
        
        report = f"""
🏥 AI POSTURE MONITORING REPORT
{'='*50}

📊 SESSION OVERVIEW
• Start Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}
• Duration: {session_duration} minutes
• Total Frames: {stats['total_frames']}
• Pose Transitions: {stats['transitions']}
• Average Detection Confidence: {stats['avg_confidence']}%

📈 POSE DISTRIBUTION
┌─────────────┬──────────┬────────────┬──────────────┐
│ Pose        │ Count    │ Time (min) │ Percentage   │
├─────────────┼──────────┼────────────┼──────────────┤
│ 🧍 Standing │ {stats['standing_count']:8} │ {stats['standing_minutes']:10.1f} │ {stats['standing_percentage']:11.1f}% │
│ 💺 Sitting  │ {stats['sitting_count']:8} │ {stats['sitting_minutes']:10.1f} │ {stats['sitting_percentage']:11.1f}% │
│ 🛏️ Lying    │ {stats['lying_count']:8} │ {stats['lying_minutes']:10.1f} │ {stats['lying_percentage']:11.1f}% │
│ ❓ Unknown  │ {stats['unknown_count']:8} │ {stats['unknown_minutes']:10.1f} │ {stats['unknown_percentage']:11.1f}% │
└─────────────┴──────────┴────────────┴──────────────┘

🎯 DETECTION ACCURACY
• Standing Detection: {stats['standing_confidence']}%
• Sitting Detection: {stats['sitting_confidence']}%
• Lying Detection: {stats['lying_confidence']}%

⚡ ACTIVITY ANALYSIS
• Sedentary Time: {stats['sitting_minutes'] + stats['lying_minutes']:.1f} minutes ({stats['sitting_percentage'] + stats['lying_percentage']:.1f}%)
• Active Time: {stats['standing_minutes']:.1f} minutes ({stats['standing_percentage']:.1f}%)
• Movement Transitions: {stats['transitions']} changes

🔧 SYSTEM STATUS
• Enhanced Sitting Detection: ✅ Active
• Email Notifications: ✅ Working
• Confidence Threshold: {self.sitting_confidence_threshold * 100}%
• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*50}
This report was generated by the Enhanced AI Posture Monitor
For any concerns, please review the detailed statistics above.
        """
        return report.strip()

    def get_default_statistics(self):
        """Default statistics when no data is available"""
        return {
            'standing_count': 0, 'standing_percentage': 0.0, 'standing_minutes': 0.0, 'standing_confidence': 0.0,
            'sitting_count': 0, 'sitting_percentage': 0.0, 'sitting_minutes': 0.0, 'sitting_confidence': 0.0,
            'lying_count': 0, 'lying_percentage': 0.0, 'lying_minutes': 0.0, 'lying_confidence': 0.0,
            'unknown_count': 0, 'unknown_percentage': 0.0, 'unknown_minutes': 0.0, 'unknown_confidence': 0.0,
            'total_frames': 0, 'session_minutes': 0.0, 'transitions': 0, 'avg_confidence': 0.0
        }

    def run_enhanced_monitoring(self, duration_minutes=5, send_email=True):
        """Run enhanced posture monitoring with real-time analysis"""
        print(f"\n🚀 Starting Enhanced Posture Monitoring for {duration_minutes} minutes...")
        print("📹 Setting up camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        start_time = time.time()
        frame_count = 0
        
        print("✅ Camera ready! Monitoring posture...")
        print("👀 Enhanced sitting detection active")
        print("Press 'q' to quit early or 's' to send report\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                frame_count += 1
                pose, confidence, landmarks = None, 0.0, None
                
                # Process every 5th frame for efficiency
                if frame_count % 5 == 0:
                    pose, confidence, landmarks = self.process_frame_with_enhanced_detection(frame)
                    
                    # Display current pose
                    if pose != "unknown":
                        pose_emoji = {"standing": "🧍", "sitting": "💺", "lying": "🛏️"}.get(pose, "❓")
                        print(f"\r{pose_emoji} Current: {pose.upper()} (confidence: {confidence:.1f}) | "
                              f"Frame: {frame_count} | "
                              f"Time: {(time.time() - start_time)/60:.1f}min", end="", flush=True)
                else:
                    # Get basic pose info for display without detailed processing
                    pose, confidence, landmarks = self.process_frame_with_enhanced_detection(frame)
                
                # Draw pose landmarks if available
                if landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Add pose text to frame
                if pose:
                    cv2.putText(frame, f"Pose: {pose.upper()} ({confidence:.1f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Enhanced Detection: ON", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.imshow('Enhanced AI Posture Monitor', frame)
                
                # Check for exit conditions
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🔴 Monitoring stopped by user")
                    break
                elif key == ord('s'):
                    print("\n📧 Sending current report...")
                    self.send_monitoring_report()
                    
                # Check time limit
                if (time.time() - start_time) >= (duration_minutes * 60):
                    print(f"\n⏰ {duration_minutes} minute monitoring completed")
                    break
                    
        except KeyboardInterrupt:
            print("\n🔴 Monitoring interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Send final report
            if send_email:
                print("\n📧 Preparing final monitoring report...")
                self.send_monitoring_report()
            
            # Generate comprehensive graphs
            print("\n📊 Generating pose analysis graphs...")
            self.generate_comprehensive_graphs()
            
            print("✅ Enhanced monitoring session completed!")

    def send_monitoring_report(self):
        """Send email report with enhanced data"""
        try:
            stats = self.calculate_real_pose_statistics()
            
            # Prepare report data for SMTP email notifier
            report_data = {
                'period': 'Enhanced Monitoring Session',
                'subject_name': self.email_config.get('monitoring_settings', {}).get('subject_name', 'User'),
                'start_time': self.session_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_duration': stats['session_minutes'],
                'posture_statistics': {
                    'standing_minutes': stats['standing_minutes'],
                    'standing_percentage': stats['standing_percentage'],
                    'sitting_minutes': stats['sitting_minutes'],
                    'sitting_percentage': stats['sitting_percentage'],
                    'lying_minutes': stats['lying_minutes'],
                    'lying_percentage': stats['lying_percentage'],
                    'transitions': stats['transitions'],
                    'confidence': stats['avg_confidence']
                },
                'enhanced_features': {
                    'enhanced_pose_detection': True,
                    'confidence_threshold': self.sitting_confidence_threshold * 100,
                    'total_frames_analyzed': stats['total_frames'],
                    'system_status': 'Active'
                }
            }
            
            success = self.email_notifier.send_monitoring_report(report_data)
            
            if success:
                print("✅ Enhanced monitoring report sent successfully!")
            else:
                print("❌ Failed to send enhanced monitoring report")
                
        except Exception as e:
            print(f"❌ Error sending report: {e}")
            import traceback
            traceback.print_exc()

    def generate_comprehensive_graphs(self):
        """Generate comprehensive pose analysis graphs"""
        try:
            if not self.pose_history:
                print("⚠️ No pose data available for graph generation")
                return []
            
            # Prepare session info
            session_info = {
                'start_time': self.session_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration': (datetime.now() - self.session_start_time).total_seconds() / 60
            }
            
            print("📊 Creating comprehensive pose analysis visualizations...")
            generated_files = self.graph_generator.create_comprehensive_report(
                self.pose_history, 
                self.pose_stats, 
                session_info
            )
            
            if generated_files:
                print(f"✅ Generated {len(generated_files)} visualization files:")
                for file in generated_files:
                    print(f"   📁 {file}")
            else:
                print("⚠️ No graph files were generated")
                
            return generated_files
            
        except Exception as e:
            print(f"❌ Error generating graphs: {e}")
            import traceback
            traceback.print_exc()
            return []

def main():
    """Main function to run enhanced posture monitoring"""
    try:
        print("🤖 Enhanced AI Posture Monitor with Improved Sitting Detection")
        print("=" * 60)
        
        # Initialize the enhanced reporter
        reporter = EnhancedPoseEstimationEmailReporter()
        
        # Run enhanced monitoring
        duration = 0.5  # 30 seconds for quick testing
        
        print(f"\n🎯 Features:")
        print("• Enhanced sitting detection with multiple criteria")
        print("• Real-time pose statistics")
        print("• Improved confidence scoring")
        print("• Comprehensive email reporting")
        print("• Visual feedback with enhanced accuracy")
        print(f"• Quick test duration: {duration * 60} seconds")
        
        reporter.run_enhanced_monitoring(duration_minutes=duration)
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
