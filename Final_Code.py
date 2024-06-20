import cv2
import cv2.typing
import mediapipe as mp

import numpy as np

from hand_landmarks import HandsLandmarks

fl = HandsLandmarks()
# cap = cv2.VideoCapture("Captured_video.mp4v")
cap = cv2.VideoCapture(0)

lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

hsvim = None

BigMargin = 50
# todo: implement small margin bounding boxes?

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
video_cod = cv2.VideoWriter_fourcc(*'XVID')
video_output = cv2.VideoWriter('Final_result.mp4', video_cod, 30, (frame_width, frame_height))

print(frame_width, frame_height)


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled



while True:
    ret, frame = cap.read()

    if (ret is True):

        frame = cv2.resize(frame, ((int)(frame.shape[1]), (int)(frame.shape[0])), fx = 0.5, fy = 0.5)

        frame_copy = frame.copy()

        # frame_copy = cv2.blur(frame_copy, (27, 27))

        height, width, _ = frame.shape

        landmarks = fl.get_hands_landmarks(frame)

        # Check hand is being detected
        if (len(landmarks) > 0):
            pt = landmarks[0]
            convexhull = cv2.convexHull(landmarks)
            boundingRect = cv2.boundingRect(landmarks)

        # Create masks
        mask = np.zeros((height, width), np.uint8)
        #SmallMask = np.zeros((height, width), np.uint8)
        hand_contour_mask = np.zeros((height, width), np.uint8)

        #cv2.polylines(frame, [convexhull], True, (0, 255, 0), 3)

        # Cehck if hand is being tracked
        if (len(landmarks) > 0):
            # cv2.fillConvexPoly(mask, convexhull, 255)
            boundingPointsWithBigMargin = np.array([[boundingRect[0] - BigMargin, boundingRect[3] + boundingRect[1] + BigMargin], [boundingRect[0] - BigMargin, boundingRect[1] - BigMargin], [boundingRect[0] + boundingRect[2] + BigMargin, boundingRect[1] - BigMargin], [boundingRect[0] + boundingRect[2] + BigMargin, boundingRect[3] + boundingRect[1] + BigMargin]], dtype=np.int32)
            cv2.fillPoly(mask, [boundingPointsWithBigMargin], 255)

            hand_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

            # Process image before defining skin HSV value
            hand_extracted_gray = cv2.cvtColor(hand_extracted, cv2.COLOR_BGR2GRAY)
            hand_extracted_edges = cv2.Canny(hand_extracted_gray, 60, 200)
            hand_extracted_blurred = cv2.GaussianBlur(hand_extracted_edges, (9, 9), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            hand_extracted_gray = clahe.apply(hand_extracted_blurred)
            hand_extracted = cv2.cvtColor(hand_extracted_gray, cv2.COLOR_GRAY2BGR)
            hsvim = cv2.cvtColor(hand_extracted, cv2.COLOR_BGR2HSV)


            # Calculate mean HSV value of hand
            landmarks = np.array([[min(frame_height - 1, landmarks[i, 1]), min(frame_width - 1, landmarks[i, 0])] for i in range(landmarks.shape[0])])
            hsvFromHand = hsvim[landmarks[:, [0]], landmarks[:, [1]]]
            hsvFromHandMean = np.array([np.mean(hsvFromHand[:, :, [0]]), np.mean(hsvFromHand[:, :, [1]]), np.mean(hsvFromHand[:, :, [2]])])

            # define the upper and lower boundaries of the HSV pixel intensities to be considered 'skin'
            HueMean = hsvFromHandMean[0]
            SaturationMean = hsvFromHandMean[1]
            ValueMean = hsvFromHandMean[2]
            lower = np.array([max(HueMean - 10, 0), max(SaturationMean - 30, 45), max(ValueMean - 50, 80)], dtype="uint8")
            upper = np.array([min(HueMean + 10, 179), min(SaturationMean + 30, 255), min(ValueMean + 50, 255)], dtype="uint8")

            skinMask = cv2.inRange(hsvim, lower, upper)

            # blur the mask to help remove noise
            skinMask = cv2.blur(skinMask, (4, 4))

            # Contour the Hands
            # gray = cv2.cvtColor(hand_extracted, cv2.COLOR_BGR2GRAY)
            # hand_extracted = cv2.cvtColor(hand_extracted, cv2.COLOR_BGRA2GRAY)

            ##threshold = cv2.adaptiveThreshold(hand_extracted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 1)
            # threshold = cv2.bitwise_and(threshold, threshold, mask=mask)

            # _, threshold = cv2.threshold(hand_extracted, 200, 255, cv2.THRESH_BINARY)


            # get threshold image
            # ret, thresh = cv2.threshold(skinMask, 150, 255, cv2.THRESH_BINARY)
            #
            # contour_hand, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # max_contour_hand = max(contour_hand, key=lambda x: cv2.contourArea(x))
            # max_contour_hand_scaled = scale_contour(max_contour_hand, 1.4)
            # # hand_extracted = cv2.cvtColor(hand_extracted, cv2.COLOR_GRAY2BGR)
            #
            # # i = 0
            # # for contour in contour_hand:
            # #     if i == 0:
            # #         i = 1
            # #         continue
            # #     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            # #     # filled_hand_contour = cv2.drawContours(hand_extracted, [contour], 0, (255, 255,255), -1)
            # #     hand_extracted = cv2.drawContours(hand_extracted, [contour], 0, (0, 255, 0), -1)
            #
            # hand_contour_mask_scaled = cv2.drawContours(hand_contour_mask, [max_contour_hand_scaled], 0, (255, 255, 255), -1)
            # hand_contour_mask = np.zeros((height, width), np.uint8)
            # hand_contour_mask = cv2.drawContours(hand_contour_mask, [max_contour_hand], 0, (255, 255, 255), -1)
            # hand_contour_mask_scaled = cv2.add(hand_contour_mask_scaled, hand_contour_mask)
            #
            # hand_masked_scaled = cv2.bitwise_and(frame_copy, frame_copy, mask = hand_contour_mask_scaled)
            # hand_masked = cv2.bitwise_and(frame_copy, frame_copy, mask = hand_contour_mask)
            #
            # # blurred_hand = cv2.GaussianBlur(hand_contour_mask, (27, 27), 0)
            # blurred_hand = cv2.GaussianBlur(hand_masked_scaled, (33,33), 7, 1, 7, borderType=cv2.BORDER_REFLECT)
            # # blurred_hand = cv2.medianBlur(hand_masked_scaled, 19)
            # blurred_hand = cv2.bitwise_and(blurred_hand, blurred_hand, mask = hand_contour_mask)



        # Extract the Back Ground

        background_mask = cv2.bitwise_not(hand_contour_mask)


        background = cv2.bitwise_and(frame, frame, mask = background_mask)


        # result = cv2.add(background, hand_extracted)
        # result = cv2.add(background, blurred_hand)
        result = hand_extracted

        # result = hand_extracted
        # result = cv2.add(background, hand_contour_mask)

        video_output.write(result)
        cv2.imshow("Final Result", result)
        # cv2.imshow("Testing", hsvim)
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


