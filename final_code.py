import cv2
import face_recognition
import os
import glob
import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
import smtplib
import winsound
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from typing import Optional, List, Dict, Tuple


class FaceRecognitionSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        casc_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(casc_path)
        self.known_faces: List = []
        self.known_names: List = []
        self.registered_faces_path = 'registered/'
        self._load_known_faces()

    def _load_known_faces(self) -> None:
        """Load all known faces from the registered directory"""
        for name in os.listdir(self.registered_faces_path):
            images_mask = f'{self.registered_faces_path}{name}/*.jpg'
            images_paths = glob.glob(images_mask)
            self.known_names += [name for _ in images_paths]
            self.known_faces += [self._get_encodings(img_path) for img_path in images_paths]

    @staticmethod
    def _get_encodings(img_path: str) -> List:
        """Get face encodings from an image file"""
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        return encodings[0] if encodings else None

    def _extract_face(self, frame) -> Optional[Tuple]:
        """Detect and extract face from frame"""
        faces = self.face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            return frame[y:y+h+50, x:x+w+50]
        return None

    def register_new_face(self, name: str) -> None:
        """Register a new face by capturing samples"""
        samples_dir = os.path.join(self.registered_faces_path, name)
        os.makedirs(samples_dir, exist_ok=True)
        
        count = 0
        while count < 50:  # Capture 50 samples
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            face = self._extract_face(frame)
            if face is not None:
                face = cv2.resize(face, (600, 600))
                file_path = os.path.join(samples_dir, f"{name}_{count}.jpg")
                cv2.imwrite(file_path, face)
                count += 1
                
                # Display count on image
                cv2.putText(face, str(count), (50, 50), 
                           cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Registering Face', face)
                
            if cv2.waitKey(1) == 13 or count == 50:  # Enter key or 50 samples
                break
                
        cv2.destroyAllWindows()
        self._load_known_faces()  # Reload known faces with new registration

    def recognize_face(self) -> Optional[str]:
        """Recognize face from webcam feed"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb)
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            face_encoding = face_recognition.face_encodings(frame_rgb, [(top, right, bottom, left)])[0]
            
            matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)
            if True in matches:
                name = self.known_names[matches.index(True)]
                cv2.putText(frame, name, (left, bottom + 20), 
                           cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.imshow('Face Recognition', frame)
                cv2.waitKey(1000)  # Show recognition for 1 second
                cv2.destroyAllWindows()
                return name
                
        return None

    def release_resources(self) -> None:
        """Release all resources"""
        self.cap.release()
        cv2.destroyAllWindows()


class VoiceAssistant:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self._configure_voice()
        self.knowledge_base = self._load_knowledge_base()

    def _configure_voice(self) -> None:
        """Configure voice properties"""
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)  # Female voice
        self.engine.setProperty('rate', 150)  # Speaking rate

    def _load_knowledge_base(self) -> Dict:
        """Load the Q&A knowledge base"""
        return {
            'time': lambda: f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}",
            'how are you': "I'm functioning optimally, thank you for asking",
            'faculty': "Faculty of Artificial Intelligence",
            'years': "The program duration is 4 years",
            'artificial intelligence': "AI is a field that combines computer science and robust datasets to enable problem-solving, encompassing machine learning and deep learning",
            'projects': "Students have created projects like smart homes, medical robots, line follower cars, and traffic light systems",
            'supervisor': "Doctor ",
            'joke': pyjokes.get_joke,
            'fields': "We have 4 fields: programming, robotics, embedded systems, and data science",
            'departments': "The General Department and Bio Artificial Intelligence Department",
            'after graduate': "Graduates receive a Bachelor's degree in Artificial Intelligence",
            'help': "You can ask me about the faculty, departments, projects, or say 'tell me a joke'"
        }

    def speak(self, text: str) -> None:
        """Convert text to speech"""
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self) -> Optional[str]:
        """Listen for and recognize speech"""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"Recognized: {command}")
                return command
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError:
                print("Could not request results from Google Speech Recognition service")
                return None
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                return None

    def search_wikipedia(self, query: str) -> None:
        """Search Wikipedia for information"""
        try:
            summary = wikipedia.summary(query, sentences=2)
            self.speak(f"According to Wikipedia: {summary}")
        except wikipedia.exceptions.DisambiguationError as e:
            self.speak(f"There are multiple options for {query}. Please be more specific.")
        except wikipedia.exceptions.PageError:
            self.speak(f"I couldn't find any information about {query}")
        except Exception as e:
            self.speak("Sorry, I encountered an error while searching Wikipedia")

    def play_video(self, query: str) -> None:
        """Play a YouTube video"""
        try:
            self.speak(f"Playing {query} on YouTube")
            pywhatkit.playonyt(query)
        except Exception as e:
            self.speak("Sorry, I couldn't play the video")
            print(f"Error playing video: {e}")

    def handle_command(self, command: str) -> bool:
        """Process user commands"""
        if not command:
            return False

        if 'sarah' in command:
            command = command.replace('sarah', '').strip()

        if 'information about' in command or 'search for' in command:
            query = command.replace('information about', '').replace('search for', '').strip()
            self.search_wikipedia(query)
            return True

        if 'play' in command and 'video' in command:
            query = command.replace('play', '').replace('video', '').strip()
            self.play_video(query)
            return True

        for key in self.knowledge_base:
            if key in command:
                response = self.knowledge_base[key]
                if callable(response):
                    response = response()
                self.speak(response)
                return True

        self.speak("I didn't understand that. Can you please repeat?")
        return False


class SecuritySystem:
    def __init__(self):
        self.email = "your_security_email@gmail.com"
        self.password = "your_app_password"  # Use app-specific password
        self.recipient = "alert_recipient@gmail.com"
        self.camera = cv2.VideoCapture(0)
        self.frame1 = None
        self.frame2 = None

    def send_alert(self, message: str) -> None:
        """Send email alert"""
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.sendmail(self.email, self.recipient, message)
                winsound.Beep(500, 1000)  # Audible alert
        except Exception as e:
            print(f"Error sending email: {e}")

    def detect_motion(self) -> None:
        """Detect motion and send alerts"""
        _, self.frame1 = self.camera.read()
        
        while True:
            _, self.frame2 = self.camera.read()
            diff = cv2.absdiff(self.frame1, self.frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(threshold, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) < 5000:
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(self.frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.send_alert("Motion detected in security camera!")
                
            cv2.imshow('Security Feed', self.frame1)
            self.frame1 = self.frame2
            
            if cv2.waitKey(1) == ord('q'):
                break
                
        self.camera.release()
        cv2.destroyAllWindows()


class ApplicationController:
    def __init__(self):
        self.face_system = FaceRecognitionSystem()
        self.voice_assistant = VoiceAssistant()
        self.security_system = SecuritySystem()

    def run_day_mode(self) -> None:
        """Run the assistant in day mode (voice and face recognition)"""
        name = self.face_system.recognize_face()
        
        if name:
            self.voice_assistant.speak(f"Hello {name}, welcome back!")
        else:
            self.voice_assistant.speak("I don't recognize you. Let's register you in the system.")
            self.voice_assistant.speak("What is your name?")
            name = self.voice_assistant.listen()
            if name:
                self.face_system.register_new_face(name)
                self.voice_assistant.speak(f"Thank you {name}. You're now registered in the system.")

        self.voice_assistant.speak("How can I assist you today?")
        
        while True:
            command = self.voice_assistant.listen()
            if command and 'exit' in command:
                self.voice_assistant.speak("Goodbye!")
                break
            self.voice_assistant.handle_command(command)

    def run_night_mode(self) -> None:
        """Run the security system in night mode"""
        self.voice_assistant.speak("Activating security mode")
        self.security_system.detect_motion()

    def run(self) -> None:
        """Main application loop with time-based mode switching"""
        try:
            current_hour = datetime.datetime.now().hour
            if 6 <= current_hour < 21:  # Daytime (6AM to 9PM)
                self.run_day_mode()
            else:  # Nighttime (9PM to 6AM)
                self.run_night_mode()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.face_system.release_resources()
            print("System shutdown complete")


if __name__ == "__main__":
    app = ApplicationController()
    app.run()