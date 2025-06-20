# Sara AI Assistant 🤖🔒  

A multi-functional AI assistant with **face recognition**, **voice interaction**, and **motion-based security alerts**.  

## Features ✨  
- **👤 Face Recognition**  
  - Register new users via webcam.  
  - Real-time recognition of known faces.  
- **🗣️ Voice Assistant (Sarah)**  
  - Answers questions (faculty info, jokes, time).  
  - Searches Wikipedia or plays YouTube videos.  
- **🔒 Security Mode**  
  - Activates at night (9 PM–6 AM).  
  - Sends email alerts on motion detection.  

## Tech Stack 🛠️  
- **Computer Vision**: OpenCV, `face_recognition`  
- **Voice**: `speech_recognition`, `pyttsx3`  
- **Security**: SMTP email alerts, motion detection  
- **Integration**: Time-based mode switching  

## Setup & Usage 🚀  
1. **Install dependencies**:  
   ```bash
   pip install opencv-python face-recognition speechrecognition pyttsx3 pywhatkit  


## Run the system:
python final_code.py  

## Commands:
- Day mode: Ask questions like "What’s the time?" or "Tell me about the faculty."
- Night mode: Automatically monitors for motion.

## Project Structure 📂
- Face Recognition: cv_version.py, data_collecter.py, face_recognition.py
- Voice Assistant: speech.py
- Security: massager_node.py
- Integrated System: final_code.py (Main entry point)

## License 📜
MIT






