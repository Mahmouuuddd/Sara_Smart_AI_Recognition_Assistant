import cv2
import numpy as np
import os



name_of_folder = input("Enter your Name: ")



# Load HAAR face classifier
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Load functions
def face_extractor(frame):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    
    
    # Crop all faces found
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        global cropped_face
        cropped_face = frame[y:y+h+50, x:x+w+50]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

parent_dir = r"C:\Users\Abdullah\Desktop\Treikaaz\registered"
path_of_newdir = os.path.join(parent_dir, name_of_folder)
the_final_path = os.mkdir(path_of_newdir)
the_final_dir = r"C:\Users\Abdullah\Desktop\Treikaaz\registered\{}".format(str(name_of_folder))
the_final_Dir = os.path.join(the_final_dir, name_of_folder)

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not False:
        count += 1
        face = cv2.resize(face_extractor(frame), (600, 600))
        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        
        #arent_dir2 = r"C:\Users\Abdullah\Desktop\FaceRecognition\Datasets\Train" + str(the_final_path)
        #the_final_path2 = os.path.join(the_final_path, name_of_folder)
        
        file_name_path = the_final_Dir  +" "+str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 50: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")