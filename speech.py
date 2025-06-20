import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime,wikipedia, pyjokes , pyfacebook
import openpyxl
import pandas as pd


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

#driver = webdriver.Chrome(ChromeDriverManager().install())




listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)

engine.say("I am sarah ")
engine.say("what do you want to ask about our faculty")


engine.runAndWait()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def talk(text):
    engine.say(text)
    engine.runAndWait()

def take_command():
    try:
        with sr.Microphone() as source:
            listener.energy_threshold=1000
            listener.adjust_for_ambient_noise(source,1.2)
            print('listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'sarah' in command:
                command = command.replace('sarah', '')
                print(command)
    except:
        pass
    return command



       
def run_sarah():
    joker = pyjokes.get_joke()
    timeNow_ = datetime.datetime.now().strftime('%I:%M %p')
    TheTimeMSE = "Current time is {}".format(timeNow_)
    
    bankOfQue = {
    'time': TheTimeMSE,
    'how are you': 'fine thank you i am ready for your question',
    'doing': 'nothing ... i am waitiing for your question',
    'faculty': 'faculty of artificial intelligance',
    'years': 'it is only 4 years ',
    'artificial intelligence': " artificial intelligence is a field, which combines computer science and robust datasets, to enable problem-solving. It also encompasses sub-fields of machine learning and deep learning",
    'projects': 'students  made a lot such as smart Homes, medical Robot, line follower cars, traffic lights ',
    'supervisor': 'Doctor ',
    'joke': joker,
    'fields': 'wa have 4 fields they are programming , robotics , embeded system and data science',
    'departments': 'The General department and bio Artificial intelligence department ',
    'after graduate': 'After the graduation The student receives Bachelors degree in Artificial Intelligence ',
    'answer' and 'dont know' and 'help': 'just say any question about the facilty and i will answer you ',
    'search': ' if you want to search of something say i want to get information',
    'thank': "your welcome"
    }
    
    command = take_command()
    print(command)
    for i in bankOfQue.keys():
        if i in command:
            talk(bankOfQue[i])
            break
        
    if 'play' and 'video' in command:
           song = command.replace('play', '')
           talk('playing ' + song)
           pywhatkit.playonyt("artificial intelligance")
    elif i not in command:
        talk("please say again")
    else:
        talk('next question')
            
def run_sara():
    class info():
      def __init__(self):
       
       self.driver = webdriver.Chrome(executable_path=r"C:/Users/basta/Desktop/chromedriver_win32/chromedriver.exe")

     
      def get_info(self,query):
        self.query=query
        self.driver.get(url="https:\\www.wikipedia.org")
        search=self.driver.find_element_by_xpath('//*[@id="searchInput"]')
        search.click()
        search.send_keys(query)
        enter = self.driver.find_element_by_xpath('//*[@id="search-form"]/fieldset/button')
        enter.click()
        say=self.driver.find_element_by_xpath('//*[@id="mw-content-text"]/div[1]/p[2]')
        talk(say.click())
        

    command = take_command()
    print(command)
    if "information" in command:
      talk("what topic you want to get information about")  
      comm=take_command()
      ass=info()
      ass.get_info(comm)
      

while True:
    run_sarah()
    run_sara()

    
