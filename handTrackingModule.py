import cv2 as cv
import mediapipe as mp
import time

class HandDetector():
    #initializing the class
    def __init__(self, static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]
        self.fingersOutput = ["zero", "one", "two", "three", "four", "five"]
    
    
    def findHands(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if (self.results.multi_hand_landmarks):
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                
        return img

    def findPosition(self, img, handNumber = 0, draw = True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber] # doing for one hand
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx = int(lm.x*w)
                cy  = int(lm.y*h)

                self.lmList.append([id, cx, cy])

                if draw:
                    cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)

        return self.lmList

    def fingersUp(self, img, draw=True):
        self.fingers = []
        if len(self.lmList) != 0:
            outputText = self.fingersOutput[0]

            if abs(self.lmList[4][1]-self.lmList[17][1]) > (self.lmList[3][1]-self.lmList[17][1]):
                self.fingers.append(1)
            else:
                self.fingers.append(0)

            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                    self.fingers.append(1)
                else:
                    self.fingers.append(0)
            
            totalFingers = self.fingers.count(1)

            if draw:
                cv.putText(img, str(self.fingersOutput[totalFingers]), (400, 120), cv.FONT_HERSHEY_DUPLEX, 5, (255, 255,255), 4)
    
        return self.fingers



def main():
    prevTime = 0
    currTime = 0
    cap = cv.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        detector.fingersUp(img)

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow("Image", img)
        cv.waitKey(1)



if __name__ == "__main__":
    main()