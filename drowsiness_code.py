#Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
import time

import board
import digitalio
import adafruit_character_lcd.character_lcd as characterlcd

import RPi.GPIO as GPIO     # Import Library to access GPIO PIN
GPIO.setmode(GPIO.BCM)    # Consider complete raspberry-pi board
GPIO.setwarnings(False)

LED_PIN = 5  
buzzer_pin = 16
lcd_rs = digitalio.DigitalInOut(board.D7)
lcd_en = digitalio.DigitalInOut(board.D8)
lcd_d7 = digitalio.DigitalInOut(board.D12)
lcd_d6 = digitalio.DigitalInOut(board.D11)
lcd_d5 = digitalio.DigitalInOut(board.D9)
lcd_d4 = digitalio.DigitalInOut(board.D10)
GPIO.setup(buzzer_pin,GPIO.OUT)   # Set pin function as output
GPIO.setup(LED_PIN,GPIO.OUT)   # Set pin function as output
# Define some device constants
GPIO.output(buzzer_pin,GPIO.LOW)

lcd_columns = 16
lcd_rows = 2

lcd = characterlcd.Character_LCD_Mono(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows)

#Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
hog_face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/pi/Downloads/shape_predictor_68_face_landmarks.dat")

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

def compute(ptA,ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a,b,c,d,e,f):
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    #Checking if it is blinked
    if(ratio>0.25):
        return 2
    elif(ratio>0.21 and ratio<=0.25):
        return 1
    else:
        return 0
    
    
lcd.cursor = True
lcd.message="welcome "#send msg to my LCD
time.sleep(2) #delay5 seconds
lcd.message="Driver Sleep\nDetection System"#send msg to my LCD
time.sleep(2) #delay



while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = hog_face_detector(gray)
        #detected face in faces array
        for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                face_frame = frame.copy()
                r = cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                #The numbers are actually the landmarks which will show eye
                left_blink = blinked(landmarks[36],landmarks[37], 
                landmarks[38], landmarks[41], landmarks[40], landmarks[39])
                right_blink = blinked(landmarks[42],landmarks[43], 
                landmarks[44], landmarks[47], landmarks[46], landmarks[45])

                #Now judge what to do for the eye blinks
                if(left_blink==0 or right_blink==0):
                        sleep+=1
                        drowsy=0
                        active=0
                        if(sleep>1):
                                status="SLEEPING !!"
                                print("SLEEPING !!")
                                GPIO.output(buzzer_pin,GPIO.HIGH)
                                GPIO.output(LED_PIN,GPIO.HIGH)
                                lcd.message='Please Wake up!'
                                time.sleep(1)
                                color = (255,0,0)
                                lcd.clear()

                elif(left_blink==1 or right_blink==1):
                        sleep=0
                        active=0
                        drowsy+=1
                        if(drowsy>1):
                                status="Drowsy :("
                                print("DROWSY :(")
                                GPIO.output(buzzer_pin,GPIO.HIGH)
                                GPIO.output(LED_PIN,GPIO.HIGH)
                                lcd.message='Dont be drowsy!'
                                time.sleep(1)
                                color = (0,0,255)
                                lcd.clear()

                else:
                        drowsy=0
                        sleep=0
                        active+=1
                        if(active>1):
                                status="Active :)"
                                print("Active :)")
                                lcd.message='All okay!\nDriver safe!'
                                time.sleep(1)
                                GPIO.output(buzzer_pin,GPIO.LOW)
                                GPIO.output(LED_PIN,GPIO.LOW)
                                color = (0,255,0)
                                lcd.clear()

                cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

                for n in range(0, 68):
                        (x,y) = landmarks[n]
                        c = cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
                        cv2.imshow("circle",c)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(2)
        if key == 27:
                break