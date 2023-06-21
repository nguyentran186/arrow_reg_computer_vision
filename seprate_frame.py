import mediapipe as mp
import numpy as np
import cv2
import os
from pynput.keyboard import Key, Controller
import time

directory = "D:\VScode\Python\ML\Hand Gesture Test"

os.chdir(directory)

kb = Controller()

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands
hands_mesh = hands.Hands(static_image_mode=True, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils
# frm = cv2.imread("hands.jpg")break

# prev_x = 0
# prev_y = 0

while True:
    _, frm = cap.read()
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    op = hands_mesh.process(rgb)


    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            print(i.landmark[8].x, i.landmark[8].y)
            if i.landmark[8].y < 1/3:
                kb.press(Key.up)
                print('Up')
            else:
                kb.release(Key.up)
            if i.landmark[8].y > 1/3*2:
                kb.press(Key.down)
                print('Down')
            else:
                kb.release(Key.down)
            if i.landmark[8].x > 1/3*2:
                kb.press(Key.left)
                print('Left')
            else:
                kb.release(Key.left)
            if i.landmark[8].x < 1/3:
                kb.press(Key.right)
                print('Right')
            else:
                kb.release(Key.right)
            
                
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=draw.DrawingSpec(
                                    color=(255, 0, 0), circle_radius=4, thickness=3),
                                connection_drawing_spec=draw.DrawingSpec(thickness=3, color=(0, 0, 255)))
    cv2.imshow("window", cv2.flip(frm, 1))
    # cv2.imwrite("hand.jpg", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
