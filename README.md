# Pose-Estimated Gym Monitoring System

[![[Project Video or GIF](https://www.facebook.com/watch/?v=2735032260077044)]](link-to-your-demo-video-or-gif) 

A real-time posture correction system using pose estimation to enhance gym workouts and promote safe exercise practices.

## Project Overview

This project addresses the challenge of incorrect exercise form, a common issue in gyms that can lead to injuries. By leveraging pose estimation technology, we've developed a system that provides users with real-time feedback on their posture during workouts. This helps individuals perform exercises correctly, reducing the risk of injury and maximizing the effectiveness of their training.

## Key Features

* **Real-time Pose Estimation:** Utilizes MediaPipe, a powerful computer vision library, to accurately detect and track key body landmarks in real time.
* **Posture Correction Feedback:** Provides immediate visual feedback on a screen, guiding users to adjust their posture for optimal form.
* **Customizable Pose Database:** Includes a database of reference poses for various exercises, allowing users to select and practice specific movements.
* **User-Friendly Interface:** Offers an intuitive interface for easy navigation and exercise selection.
* **Future Enhancement:** Planned integration of wireless headset feedback for a more immersive and hands-free experience.

## Technical Details

* **Programming Language:** Python
* **Libraries:** MediaPipe, OpenCV
* **Dataset:** Custom pose dataset focused on Bangladeshi and Indian Subcontinent ethnicities.

## How It Works

1. **Capture:** The system captures video input from a smartphone or computer camera.
2. **Pose Detection:** MediaPipe's pose estimation model identifies key body joints and their coordinates.
3. **Geometric Analysis:** The system compares the detected pose with reference poses from the database.
4. **Feedback:** Real-time feedback is provided on the screen, highlighting any deviations from the correct posture.

## Getting Started

1. **Clone the Repository:** `git clone https://github.com/1yakub/PoseEstimationProject`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run the Application:** `python main.py`

## Future Enhancements

* **3D Pose Estimation:** Incorporate depth data for more accurate posture analysis.
* **Exercise Recommendations:** Suggest exercises based on individual fitness goals and progress.
* **Gamification:** Introduce gamified elements to make workouts more engaging.

## Project Demo

[[Include a link to a video demonstrating your project in action](https://github.com/1yakub/PoseEstimationProject/blob/master/AITrainer/aitrainer-demo.jpg)]

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes.

## Contact

Md. Yakub Hossain - yakub7788@gmail.com
