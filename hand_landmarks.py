# https://pysource.com/blur-faces-in-real-time-with-opencv-mediapipe-and-python
import mediapipe as mp
import cv2
import numpy as np


class HandsLandmarks:
    def __init__(self):
        mp_hand_mesh = mp.solutions.hands
        self.hands = mp_hand_mesh.Hands()

    def get_hands_landmarks(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        handlandmarks = []
        if result.multi_hand_landmarks is not None:
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(0, 20):
                    pt1 = hand_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    handlandmarks.append([x, y])
        return np.array(handlandmarks, np.int32)