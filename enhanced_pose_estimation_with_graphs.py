"""
Enhanced AI Posture Monitor with Improved Sitting Detection, Email Reporting, and Real-time Graphs
Uses MediaPipe with enhanced detection thresholds, real-time email reporting, and dynamic plotting
"""
import cv2
import time
import json
from datetime import datetime, timedelta
import os
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading

# Import the pose estimation package and email system
from ai_posture_monitor_package.ai_posture_monitor.pose_est_dependencies import PoseEstimation
from smtp_email_system import SMTPEmailNotifier

class EnhancedPoseEstimationWithGraphs:
    def __init__(self, config_file='email_config.json'):
        """Initialize with email configuration and graphing capabilities"""
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
        
        # Enhanced pose tracking
        self.pose_history = []
        self.last_pose = None
        self.pose_transition_count = 0
        self.session_start_time = datetime.now()
        self.monitoring_active = True
        
        # Pose counters with detailed tracking
        self.pose_stats = {
            'standing': {'count': 0, 'duration': 0, 'confidence_scores': []},
            'sitting': {'count': 0, 'duration': 0, 'confidence_scores': []},
            'lying': {'count': 0, 'duration': 0, 'confidence_scores': []},
            'unknown': {'count': 0, 'duration': 0, 'confidence_scores': []}
        }
        
        # Enhanced sitting detection parameters
        self.sitting_confidence_threshold = 0.5  # Lowered for better sensitivity
        self.sitting_confirmation_frames = 3     # Reduced for faster detection
        self.recent_sitting_predictions = []
        
        # Real-time graphing data
        self.graph_data = {
            'timestamps': deque(maxlen=100),
            'standing': deque(maxlen=100),
            'sitting': deque(maxlen=100),
            'lying': deque(maxlen=100),
            'confidence': deque(maxlen=100),
            'transitions': deque(maxlen=100)
        }
        
        # Graph setup
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Enhanced AI Posture Monitor - Real-time Analysis', fontsize=14, fontweight='bold')
        plt.ion()  # Interactive mode on
        
        print("✅ Enhanced Posture Monitor initialized with improved sitting detection and graphing")

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

    def enhanced_sitting_detection(self, pose_landmarks, frame_confidence):
        """Enhanced sitting detection with multiple criteria"""
        if not pose_landmarks:
            return "unknown", 0.0
            
        try:
            # Get key landmarks
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
                pose, confidence = self.enhanced_sitting_detection(results.pose_landmarks, frame_confidence)
                
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
        """Update pose statistics with real data and graph data"""
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
        
        # Update graph data
        current_time = (datetime.now() - self.session_start_time).total_seconds() / 60
        self.graph_data['timestamps'].append(current_time)
        
        # Add pose data (1 for active pose, 0 for inactive)
        self.graph_data['standing'].append(1 if pose == 'standing' else 0)
        self.graph_data['sitting'].append(1 if pose == 'sitting' else 0)
        self.graph_data['lying'].append(1 if pose == 'lying' else 0)
        self.graph_data['confidence'].append(confidence * 100)
        self.graph_data['transitions'].append(self.pose_transition_count)

    def update_graphs(self):
        """Update real-time graphs"""
        try:
            # Clear all subplots
            for ax in self.axes.flat:
                ax.clear()
            
            if len(self.graph_data['timestamps']) < 2:
                return
            
            times = list(self.graph_data['timestamps'])
            
            # Graph 1: Pose Detection Over Time
            ax1 = self.axes[0, 0]
            ax1.plot(times, list(self.graph_data['standing']), 'g-', label='Standing', linewidth=2)
            ax1.plot(times, list(self.graph_data['sitting']), 'b-', label='Sitting', linewidth=2)
            ax1.plot(times, list(self.graph_data['lying']), 'r-', label='Lying', linewidth=2)
            ax1.set_title('Pose Detection Over Time', fontweight='bold')
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Pose Active (1=Yes, 0=No)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.1, 1.1)
            
            # Graph 2: Confidence Levels
            ax2 = self.axes[0, 1]
            confidence_values = list(self.graph_data['confidence'])
            ax2.plot(times, confidence_values, 'purple', linewidth=2)
            ax2.fill_between(times, confidence_values, alpha=0.3, color='purple')
            ax2.set_title('Detection Confidence', fontweight='bold')
            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Confidence (%)')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            
            # Graph 3: Cumulative Pose Distribution
            ax3 = self.axes[1, 0]
            stats = self.calculate_real_pose_statistics()
            poses = ['Standing', 'Sitting', 'Lying', 'Unknown']
            percentages = [
                stats['standing_percentage'], 
                stats['sitting_percentage'],
                stats['lying_percentage'],
                stats['unknown_percentage']
            ]
            colors = ['green', 'blue', 'red', 'gray']
            
            wedges, texts, autotexts = ax3.pie(percentages, labels=poses, colors=colors, 
                                               autopct='%1.1f%%', startangle=90)
            ax3.set_title('Pose Distribution', fontweight='bold')
            
            # Graph 4: Transitions Over Time
            ax4 = self.axes[1, 1]
            transitions = list(self.graph_data['transitions'])
            ax4.plot(times, transitions, 'orange', linewidth=2, marker='o', markersize=3)
            ax4.set_title('Pose Transitions', fontweight='bold')
            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('Total Transitions')
            ax4.grid(True, alpha=0.3)
            
            # Add session info
            session_info = f"Session Time: {stats['session_minutes']:.1f}min | Frames: {stats['total_frames']} | Avg Confidence: {stats['avg_confidence']:.1f}%"
            self.fig.suptitle(f'Enhanced AI Posture Monitor - Real-time Analysis\n{session_info}', 
                            fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"⚠️ Error updating graphs: {e}")

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

    def get_default_statistics(self):
        """Default statistics when no data is available"""
        return {
            'standing_count': 0, 'standing_percentage': 0.0, 'standing_minutes': 0.0, 'standing_confidence': 0.0,
            'sitting_count': 0, 'sitting_percentage': 0.0, 'sitting_minutes': 0.0, 'sitting_confidence': 0.0,
            'lying_count': 0, 'lying_percentage': 0.0, 'lying_minutes': 0.0, 'lying_confidence': 0.0,
            'unknown_count': 0, 'unknown_percentage': 0.0, 'unknown_minutes': 0.0, 'unknown_confidence': 0.0,
            'total_frames': 0, 'session_minutes': 0.0, 'transitions': 0, 'avg_confidence': 0.0
        }

    def save_session_graphs(self):
        """Save the final graphs as images"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"posture_analysis_{timestamp}.png"
            filepath = os.path.join(os.getcwd(), filename)
            
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"📊 Session graphs saved as: {filename}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving graphs: {e}")
            return None

    def run_enhanced_monitoring_with_graphs(self, duration_minutes=5, send_email=True):
        """Run enhanced posture monitoring with real-time analysis and graphs"""
        print(f"\n🚀 Starting Enhanced Posture Monitoring with Graphs for {duration_minutes} minutes...")
        print("📹 Setting up camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        start_time = time.time()
        frame_count = 0
        last_graph_update = time.time()
        
        print("✅ Camera ready! Monitoring posture with real-time graphs...")
        print("👀 Enhanced sitting detection active")
        print("📊 Real-time graphs updating...")
        print("Press 'q' to quit early or 's' to send report\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                frame_count += 1
                pose, confidence, landmarks = None, 0.0, None
                
                # Process every 3rd frame for efficiency
                if frame_count % 3 == 0:
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
                
                # Update graphs every 2 seconds
                if time.time() - last_graph_update > 2.0:
                    self.update_graphs()
                    last_graph_update = time.time()
                
                # Draw pose landmarks if available
                if landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Add pose text to frame
                if pose:
                    cv2.putText(frame, f"Pose: {pose.upper()} ({confidence:.1f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Enhanced Detection + Graphs: ON", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.imshow('Enhanced AI Posture Monitor with Graphs', frame)
                
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
            
            # Final graph update
            self.update_graphs()
            
            # Save session graphs
            graph_file = self.save_session_graphs()
            
            # Send final report
            if send_email:
                print("\n📧 Preparing final monitoring report...")
                self.send_monitoring_report()
            
            print("✅ Enhanced monitoring session with graphs completed!")
            print("📊 Graphs window will remain open - close manually when done reviewing")

    def send_monitoring_report(self):
        """Send email report with enhanced data"""
        try:
            stats = self.calculate_real_pose_statistics()
            
            # Prepare report data for SMTP email notifier
            report_data = {
                'period': 'Enhanced Monitoring Session with Graphs',
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
                    'enhanced_sitting_detection': True,
                    'real_time_graphs': True,
                    'confidence_threshold': self.sitting_confidence_threshold * 100,
                    'total_frames_analyzed': stats['total_frames'],
                    'system_status': 'Active with Graphing'
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

def main():
    """Main function to run enhanced posture monitoring with graphs"""
    try:
        print("🤖 Enhanced AI Posture Monitor with Improved Sitting Detection + Real-time Graphs")
        print("=" * 80)
        
        # Initialize the enhanced reporter with graphs
        reporter = EnhancedPoseEstimationWithGraphs()
        
        # Run enhanced monitoring
        duration = 0.5  # 30 seconds for quick testing
        
        print(f"\n🎯 Features:")
        print("• Enhanced sitting detection with multiple criteria")
        print("• Real-time pose statistics")
        print("• Improved confidence scoring")
        print("• Comprehensive email reporting")
        print("• Visual feedback with enhanced accuracy")
        print("• 📊 Real-time graphing and analysis")
        print("• 🖼️ Session graph saving")
        print(f"• Quick test duration: {duration * 60} seconds")
        
        reporter.run_enhanced_monitoring_with_graphs(duration_minutes=duration)
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
