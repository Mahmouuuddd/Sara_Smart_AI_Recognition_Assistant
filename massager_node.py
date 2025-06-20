import cv2
import smtplib # simple message trasport portcol
import winsound
from email.message import EmailMessage

def mail_alart(to,text_mail):
        server=smtplib.SMTP("smtp.gmail.com",587)
        server.starttls()
        server.login() #put your Gmail
        server.sendmail('', to , text_mail) #also here
        winsound.Beep(500, 200)
cam=cv2.VideoCapture(0)
while cam.isOpened():
    ret, frame1= cam.read()
    ret, frame2= cam.read()
    diff=cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        mail_alart('', 'There is something wrong')
    if cv2.waitKey(1)==ord('q'):
        break
    cv2.imshow('cv',frame1)
