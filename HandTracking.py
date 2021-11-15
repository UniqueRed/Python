#Import essentials
import cv2
import mediapipe as mp
import time as t

#Define the camera
cam = cv2.VideoCapture(0)

#Get the hand data from mediapipe in order for the AI to know what a hand looks like
handData = mp.solutions.hands
hands = handData.Hands(max_num_hands = 2)
draw = mp.solutions.drawing_utils

#Start a while loop in order to keep the image running along with the skeletal overlay
while True:
    success, img = cam.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Compute the position of the hands
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks)

    #Checks for multiple hands
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                #Convert floats to pixel values to accurately track the position of each landmark
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, ":", cx, cy)

                #Detect certain landmarks and identify them with different colors
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                if id == 12:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                if id == 16:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                if id == 20:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

            #Draw the skeleton overlay on the hands
            draw.draw_landmarks(img, handLms, handData.HAND_CONNECTIONS)

    #Flip the image
    img = cv2.flip(img, 1)
    
    #Display on the camera
    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)
