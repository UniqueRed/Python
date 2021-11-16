#Import essentials
import cv2
import mediapipe as mp

#Create a class
class handDetector():
    def __init__(self, mode = False, maxHands = 1, complexity = 1, detectCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectCon = detectCon
        self.trackCon = trackCon

        #Get the hand data from mediapipe in order for the AI to know what a hand looks like
        self.handData = mp.solutions.hands
        self.hands = self.handData.Hands(self.mode, self.maxHands, self.complexity, self.detectCon, self.trackCon)
        self.draw = mp.solutions.drawing_utils

    def trackHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #Compute the position of the hands
        self.result = self.hands.process(imgRGB)

        #Checks for multiple hands
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    #Draw the skeleton overlay on the hands
                    self.draw.draw_landmarks(img, handLms, self.handData.HAND_CONNECTIONS)
        return img
    
    def findPos(self, img, handNo = 0, draw = True):
    #Create a list in order to keep track of the landmarks
        lmList = []
        
        #Check for details about the hands
        if self.result.multi_hand_landmarks:
            hand = self.result.multi_hand_landmarks[handNo]
        
            for id, lm in enumerate(hand.landmark):
                #Convert floats to pixel values to accurately track the position of each landmark
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                #Draw extra dots on the landmarks for better visualization
                if draw:
                    cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
        return lmList

def main():
    #Define the camera and set it's size
    camW, camH = 1080, 720

    cam = cv2.VideoCapture(0)
    cam.set(3, camW)
    cam.set(4, camH)

    #Reference the class
    detector = handDetector()
    
    while True:
        success, img = cam.read()
        img = detector.trackHands(img)

        #Find and print the position of landmarks
        lmList = detector.findPos(img)
        if len(lmList) != 0:
            print(lmList[8])

        #Flip the image
        img = cv2.flip(img, 1)

        #Display on the camera
        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
