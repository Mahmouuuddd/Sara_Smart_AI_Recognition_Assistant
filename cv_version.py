import cv2
import face_recognition
import os
import glob

cap = cv2.VideoCapture(0)

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


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
        global cropped_face
        cropped_face = frame[y:y+h+50, x:x+w+50]

    return cropped_face

count = 0

def Make_Folder():
    name_of_folder = input("Enter your Name: ")
    #Enter your path of the computer
    parent_dir = r"C:\Users\Abdullah\Desktop\Treikaaz\registered"
    path_of_newdir = os.path.join(parent_dir, name_of_folder)
    the_final_path = os.mkdir(path_of_newdir)
    the_final_dir = r"C:\Users\Abdullah\Desktop\Treikaaz\registered\{}".format(str(name_of_folder))
    global the_final_Dir
    the_final_Dir = os.path.join(the_final_dir, name_of_folder)    
    
    return the_final_Dir



known_faces = []
known_names = []
known_faces_paths = []

registered_faces_path = 'registered/'
for name in os.listdir(registered_faces_path):
    images_mask = '%s%s/*.jpg' % (registered_faces_path, name)
    images_paths = glob.glob(images_mask) 
    known_faces_paths += images_paths
    known_names += [name for x in images_paths]

def get_encodings(img_path):
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)
    return encoding[0]

known_faces = [get_encodings(img_path) for img_path in known_faces_paths]


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(frame_rgb)
    for face in faces: # top, right, bottom, left
        top, right, bottom, left = face
        cv2.rectangle(frame, (left, top), (right, bottom),(0,0,255), 2)
        face_code = face_recognition.face_encodings(frame_rgb, [face])[0]

        results = face_recognition.compare_faces(known_faces, face_code, tolerance=0.6)
        if any(results):
            name = known_names[results.index(True)]
        else:
            Make_Folder()
            while True:
            
                ret, frame = cap.read()
                if face_extractor(frame) is not False:
                    count += 1
                    face = cv2.resize(face_extractor(frame), (600, 600))
                                        
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
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

    cv2.imshow('Sarah', frame)
    k = cv2.waitKey(1)
    if ord('q') == k:
        break
cv2.destroyAllWindows()
cap.release()
