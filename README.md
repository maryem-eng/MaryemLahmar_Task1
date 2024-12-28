# MaryemLahmar_Task1

*Instructions for Setting Up and Running the Code
Before running the code, you need to install the following libraries: 
- OpenCV for video capture and image processing: pip install opencv-python
- MediaPipe for face and eye detection: pip install mediapipe
- NumPy for data manipulation and mathematical calculations: pip install numpy
- Scipy for calculating the Euclidean distance used in the eye aspect ratio calculation: pip install scipy
- Winsound (for Windows only) to play an alarm sound on Windows systems when fatigue is detected: pip install winsound
To set up, ensure your camera is connected and working on your machine. Clone the repository to your local machine: git clone https://github.com/maryem-eng/MaryemLahmar_Task1.git
To run the code, execute the Python file code.py with: python code.py
The program will start capturing frames from the camera and perform eye blink and fatigue detection. If fatigue is detected, an alert message will appear, and an alarm sound will be played.
To quit the program, press the 'q' key to exit the detection application.

*Assumptions
-Single Face Detection: The program is designed to detect only one face at a time. If multiple faces are present, it may not function as expected.

*Challenges Faced
-Eye Detection Accuracy: Eye detections are not always perfectly accurate, especially in low light conditions or if the person is wearing glasses. This can affect the accuracy of the Eye Aspect Ratio (EAR) 
 calculation.
