import cv2
import mediapipe as mp
import time

#getting video from webcam
cap = cv2.VideoCapture(0)

#detecting the hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    #reading and opening video in window
    success, img = cap.read()
    #img processing
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #checking how many hands are there
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark): #getting position values of the landmarks 
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                '''print(id, cx, cy)
                if id == 0: #singling out a single point
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)'''
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #drawing the 21 points on the hand with lines


    #fixing FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3) #display on screen

    cv2.imshow('image', img)
    cv2.waitKey(1)
