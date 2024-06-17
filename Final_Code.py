import cv2
import cv2.typing
import mediapipe as mp

import numpy as np

from hand_landmarks import HandsLandmarks

fl = HandsLandmarks()
cap = cv2.VideoCapture("Captured_video.mp4v")

lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

BigMargin = 50
SmallMargin = 20

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

        # frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
        frame = cv2.resize(frame, ((int)(frame.shape[1]), (int)(frame.shape[0])), fx = 0.5, fy = 0.5)

        frame_copy = frame.copy()

        # frame_copy = cv2.blur(frame_copy, (27, 27))

        height, width, _ = frame.shape

        landmarks = fl.get_hands_landmarks(frame)

        # print(len(landmarks))
        if (len(landmarks) > 0):
            pt = landmarks[0]

            convexhull = cv2.convexHull(landmarks)
            boundingRect = cv2.boundingRect(landmarks)



        mask = np.zeros((height, width), np.uint8)
        SmallMask = np.zeros((height, width), np.uint8)

        hand_contour_mask = np.zeros((height, width), np.uint8)

        #cv2.polylines(frame, [convexhull], True, (0, 255, 0), 3)

        if (len(landmarks) > 0):
            # cv2.fillConvexPoly(mask, convexhull, 255)
            boundingPointsWithBigMargin = np.array([[boundingRect[0] - BigMargin, boundingRect[3] + boundingRect[1] + BigMargin], [boundingRect[0] - BigMargin, boundingRect[1] - BigMargin], [boundingRect[0] + boundingRect[2] + BigMargin, boundingRect[1] - BigMargin], [boundingRect[0] + boundingRect[2] + BigMargin, boundingRect[3] + boundingRect[1] + BigMargin]], dtype=np.int32)
            boundingPointsWithSmallMargin = np.array([[boundingRect[0] - SmallMargin, boundingRect[3] + boundingRect[1] + SmallMargin], [boundingRect[0] - SmallMargin, boundingRect[1] - SmallMargin], [boundingRect[0] + boundingRect[2] + SmallMargin, boundingRect[1] - SmallMargin], [boundingRect[0] + boundingRect[2] + SmallMargin, boundingRect[3] + boundingRect[1] + SmallMargin]], dtype=np.int32)
            cv2.fillPoly(mask, [boundingPointsWithBigMargin], 255)
            cv2.fillPoly(SmallMask, [boundingPointsWithSmallMargin], 255)


            # cv2.fillPoly(mask, [cv2.typing.Point2f(boundingRect[0], boundingRect[1]), cv2.typing.Point2f(boundingRect[2], boundingRect[3])], 255)
            # cv2.fillPoly(mask, np.int8(np.array([[boundingRect[0], boundingRect[1]], [boundingRect[2], boundingRect[3]]])), 255)

            hand_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask = mask)

            # Blurring the Hand

            # blurred_hand = cv2.GaussianBlur(hand_extracted, (27, 27), 0)

            # define the upper and lower boundaries of the HSV pixel intensities
            # to be considered 'skin'
            hsvim = cv2.cvtColor(hand_extracted, cv2.COLOR_BGR2HSV)

            skinMask = cv2.inRange(hsvim, lower, upper)

            # blur the mask to help remove noise
            skinMask = cv2.blur(skinMask, (4, 4))

            # Contour the Hands
            # gray = cv2.cvtColor(hand_extracted, cv2.COLOR_BGR2GRAY)
            hand_extracted = cv2.cvtColor(hand_extracted, cv2.COLOR_BGRA2GRAY)

            ##threshold = cv2.adaptiveThreshold(hand_extracted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 1)
            # threshold = cv2.bitwise_and(threshold, threshold, mask=mask)
            _, threshold = cv2.threshold(hand_extracted, 200, 255, cv2.THRESH_BINARY)


            # get threshold image
            ret, thresh = cv2.threshold(skinMask, 150, 255, cv2.THRESH_BINARY)

            contour_hand, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            max_contour_hand = max(contour_hand, key=lambda x: cv2.contourArea(x))
            max_contour_hand_scaled = scale_contour(max_contour_hand, 1.3)
            # max_contour_hand = max(contour_hand, key=cv2.contourArea)
            hand_extracted = cv2.cvtColor(hand_extracted, cv2.COLOR_GRAY2RGB)

            # i = 0
            # for contour in contour_hand:
            #     if i == 0:
            #         i = 1
            #         continue
            #     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            #     # filled_hand_contour = cv2.drawContours(hand_extracted, [contour], 0, (255, 255,255), -1)
            #     hand_extracted = cv2.drawContours(hand_extracted, [contour], 0, (0, 255, 0), -1)

            # hand_extracted = cv2.drawContours(hand_extracted, [max_contour_hand], -1, (255, 255, 255), -1)
            hand_contour_mask_scaled = cv2.drawContours(hand_contour_mask, [max_contour_hand_scaled], 0, (255, 255, 255), -1)
            hand_contour_mask = cv2.drawContours(hand_contour_mask, [max_contour_hand], 0, (255, 255, 255), -1)

            hand_masked = cv2.bitwise_and(frame_copy, frame_copy, mask = hand_contour_mask_scaled)

            # blurred_hand = cv2.GaussianBlur(hand_contour_mask, (27, 27), 0)
            blurred_hand = cv2.GaussianBlur(hand_masked, (27, 27), 1, 1, 1, borderType=cv2.BORDER_REPLICATE)

            # hand_masked = scale_image(hand_masked, 0.7)


        # Extract the Back Ground

        background_mask = cv2.bitwise_not(hand_contour_mask)


        background = cv2.bitwise_and(frame, frame, mask = background_mask)


        # result = cv2.add(background, hand_extracted)
        result = cv2.add(background, blurred_hand)
        # result = background
        # result = hand_extracted
        # result = cv2.add(background, hand_contour_mask)

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


