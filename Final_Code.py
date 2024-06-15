import cv2
import mediapipe as mp

import numpy as np

from hand_landmarks import HandsLandmarks

fl = HandsLandmarks()
cap = cv2.VideoCapture("Captured_video.mp4v")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
video_cod = cv2.VideoWriter_fourcc(*'XVID')
video_output = cv2.VideoWriter('Final_result.mp4', video_cod, 30, (frame_width, frame_height))

print(frame_width, frame_height)


while True:
    ret, frame = cap.read()

    if (ret is True):

        # frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
        frame = cv2.resize(frame, ((int)(frame.shape[1]), (int)(frame.shape[0])), fx = 0.5, fy = 0.5)

        frame_copy = frame.copy()

        frame_copy = cv2.blur(frame_copy, (27, 27))

        height, width, _ = frame.shape

        landmarks = fl.get_hands_landmarks(frame)

        # print(len(landmarks))
        if (len(landmarks) > 0):
            pt = landmarks[0]

            convexhull = cv2.convexHull(landmarks)

        mask = np.zeros((height, width), np.uint8)
        #cv2.polylines(frame, [convexhull], True, (0, 255, 0), 3)

        if (len(landmarks) > 0):
            cv2.fillConvexPoly(mask, convexhull, 255)

            hand_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask = mask)
            # Blurring the Face

            blurred_hand = cv2.GaussianBlur(hand_extracted, (27, 27), 0)


        # Extract the Back Ground

        background_mask = cv2.bitwise_not(mask)


        background = cv2.bitwise_and(frame, frame, mask = background_mask)


        result = cv2.add(background, hand_extracted)

        video_output.write(result)
        cv2.imshow("Final Result", result)

        #cv2.imshow("Background", background)


        #cv2.imshow("Mask Inverse", background_mask)
        #cv2.imshow("Blurred Face", blurred_hand)

        cv2.imshow("Frame", frame)

        #cv2.imshow("Face Extracted", hand_extracted)

        #cv2.imshow("Mask", mask)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_output.release()

cv2.destroyAllWindows()


