import cv2
import cv2.typing
import mediapipe as mp

import numpy as np
import matplotlib as mpl

from hand_landmarks import HandsLandmarks

fl = HandsLandmarks()
# cap = cv2.VideoCapture("Captured_video.mp4v")
cap = cv2.VideoCapture(0)

lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

hsvim = None
skin_mask = None

big_margin = 50
small_margin = 20
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
            convex_hull = cv2.convexHull(landmarks)
            bounding_rect = cv2.boundingRect(landmarks)

        # Create masks
        big_mask = np.zeros((height, width), np.uint8)
        small_mask = np.zeros((height, width), np.uint8)
        hand_contour_mask = np.zeros((height, width), np.uint8)

        #cv2.polylines(frame, [convexhull], True, (0, 255, 0), 3)

        # Cehck if hand is being tracked
        if (len(landmarks) > 0):
            # cv2.fillConvexPoly(mask, convexhull, 255)
            bounding_points_with_big_margin = np.array([[bounding_rect[0] - big_margin, bounding_rect[3] + bounding_rect[1] + big_margin], [bounding_rect[0] - big_margin, bounding_rect[1] - big_margin], [bounding_rect[0] + bounding_rect[2] + big_margin, bounding_rect[1] - big_margin], [bounding_rect[0] + bounding_rect[2] + big_margin, bounding_rect[3] + bounding_rect[1] + big_margin]], dtype=np.int32)
            bounding_points_with_small_margin = np.array([[bounding_rect[0] - small_margin, bounding_rect[3] + bounding_rect[1] + small_margin], [bounding_rect[0] - small_margin, bounding_rect[1] - small_margin], [bounding_rect[0] + bounding_rect[2] + small_margin, bounding_rect[1] - small_margin], [bounding_rect[0] + bounding_rect[2] + small_margin, bounding_rect[3] + bounding_rect[1] + small_margin]], dtype=np.int32)

            cv2.fillPoly(big_mask, [bounding_points_with_big_margin], 255)
            cv2.fillPoly(small_mask, [bounding_points_with_small_margin], 255)

            hand_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=big_mask)

            # Process image before defining skin HSV value
            hand_extracted_gray = cv2.cvtColor(hand_extracted, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            hand_extracted_CLAHE_processed = clahe.apply(hand_extracted_gray)
            hand_extracted_blurred = cv2.GaussianBlur(hand_extracted_CLAHE_processed, (9, 9), 0)
            hand_extracted_edges = cv2.Canny(hand_extracted_blurred, 80, 150)
            hand_extracted_edges = cv2.dilate(hand_extracted_edges, np.ones((3,3),np.uint8), iterations=2)
            hand_extracted_edges = cv2.morphologyEx(hand_extracted_edges, cv2.MORPH_GRADIENT, np.ones((3,3),np.uint8))
            hand_extracted_edges_processed = cv2.bitwise_and(hand_extracted_edges, hand_extracted_edges, mask = small_mask)
            # hand_extracted = cv2.cvtColor(hand_extracted_gray, cv2.COLOR_GRAY2BGR)
            hsvim = cv2.cvtColor(hand_extracted, cv2.COLOR_BGR2HSV)


            # Calculate mean HSV value of hand
            landmarks = np.array([[min(frame_height - 1, landmarks[i, 1]), min(frame_width - 1, landmarks[i, 0])] for i in range(landmarks.shape[0])])
            hsvFromHand = hsvim[landmarks[:, [0]], landmarks[:, [1]]]
            hsvFromHandMean = np.array([np.mean(hsvFromHand[:, :, [0]]), np.mean(hsvFromHand[:, :, [1]]), np.mean(hsvFromHand[:, :, [2]])])

            # define the upper and lower boundaries of the HSV pixel intensities to be considered 'skin'
            HueMean = hsvFromHandMean[0]
            SaturationMean = hsvFromHandMean[1]
            ValueMean = hsvFromHandMean[2]
            lower = np.array([max(HueMean - 40, 0), max(SaturationMean - 40, 45), max(ValueMean - 50, 80)], dtype="uint8")
            upper = np.array([min(HueMean + 40, 179), min(SaturationMean + 40, 255), min(ValueMean + 50, 255)], dtype="uint8")

            skin_mask = cv2.inRange(hsvim, lower, upper)

            print(hsvFromHandMean)

            # blur the mask to help remove noise
            skin_mask = cv2.blur(skin_mask, (4, 4))

            skin_mask = cv2.add(skin_mask, hand_extracted_edges_processed)
            hand_extracted_edges_processed_invert = cv2.bitwise_not(hand_extracted_edges_processed)
            skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=hand_extracted_edges_processed_invert)

            # Contour the Hands
            # gray = cv2.cvtColor(hand_extracted, cv2.COLOR_BGR2GRAY)
            # hand_extracted = cv2.cvtColor(hand_extracted, cv2.COLOR_BGRA2GRAY)

            ##threshold = cv2.adaptiveThreshold(hand_extracted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 1)
            # threshold = cv2.bitwise_and(threshold, threshold, mask=mask)

            # _, threshold = cv2.threshold(hand_extracted, 200, 255, cv2.THRESH_BINARY)


            contour_processed_hand, _ = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            max_contour_processed_hand = max(contour_processed_hand, key=lambda x: cv2.contourArea(x))

            # i = 0
            # for contour in contour_processed_hand:
            #     if i == 0:
            #         i = 1
            #         continue
            #     # approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            #     filled_hand_contour = cv2.drawContours(hand_extracted, [contour], 0, (255, 255,255), -1)
            #     # hand_extracted = cv2.drawContours(hand_extracted, [contour], 0, (0, 255, 0), -1)

            # get threshold image
            ret, thresh = cv2.threshold(skin_mask, 150, 255, cv2.THRESH_BINARY)

            contour_hand, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            max_contour_hand = max(contour_hand, key=lambda x: cv2.contourArea(x))
            max_contour_hand_scaled = scale_contour(max_contour_hand, 1.4)
            # hand_extracted = cv2.cvtColor(hand_extracted, cv2.COLOR_GRAY2BGR)

            # i = 0
            # for contour in contour_hand:
            #     if i == 0:
            #         i = 1
            #         continue
            #     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            #     # filled_hand_contour = cv2.drawContours(hand_extracted, [contour], 0, (255, 255,255), -1)
            #     hand_extracted = cv2.drawContours(hand_extracted, [contour], 0, (0, 255, 0), -1)

            ### hand_contour_mask_scaled = cv2.drawContours(hand_contour_mask, [max_contour_hand_scaled], 0, (255, 255, 255), -1)
            hand_contour_mask_scaled = cv2.drawContours(hand_contour_mask, [max_contour_processed_hand], 0, (255, 255, 255), -1)
            hand_contour_mask = np.zeros((height, width), np.uint8)
            ### hand_contour_mask = cv2.drawContours(hand_contour_mask, [max_contour_hand], 0, (255, 255, 255), -1)
            hand_contour_mask = cv2.drawContours(hand_contour_mask, [max_contour_processed_hand], 0, (255, 255, 255), -1)
            hand_contour_mask_scaled = cv2.add(hand_contour_mask_scaled, hand_contour_mask)

            hand_masked_scaled = cv2.bitwise_and(frame_copy, frame_copy, mask = hand_contour_mask_scaled)
            hand_masked = cv2.bitwise_and(frame_copy, frame_copy, mask = hand_contour_mask)

            # blurred_hand = cv2.GaussianBlur(hand_contour_mask, (27, 27), 0)
            blurred_hand = cv2.GaussianBlur(hand_masked_scaled, (33,33), 7, 1, 7, borderType=cv2.BORDER_REFLECT)
            blurred_hand = cv2.GaussianBlur(hand_masked_scaled, (33,33), 7, 1, 7, borderType=cv2.BORDER_REFLECT)
            # blurred_hand = cv2.medianBlur(hand_masked_scaled, 19)
            blurred_hand = cv2.bitwise_and(blurred_hand, blurred_hand, mask = hand_contour_mask)



        # Extract the Back Ground

        background_mask = cv2.bitwise_not(hand_contour_mask)


        background = cv2.bitwise_and(frame, frame, mask = background_mask)


        # result = cv2.add(background, hand_extracted)
        # result = cv2.add(background, blurred_hand)
        result = skin_mask

        # result = hand_extracted
        # result = cv2.add(background, hand_contour_mask)

        video_output.write(result)
        cv2.imshow("Final Result", blurred_hand)
        cv2.imshow("edges", hand_extracted_edges_processed)
        cv2.imshow("skin mask", skin_mask)
        cv2.imshow("hsvim", hsvim)
        mpl.colors.hsv_to_rgb(hsvFromHandMean)
        handcolor = cv2.rectangle(hsvim, (0,0), (frame.shape[1], frame.shape[0]), color=hsvFromHandMean, thickness=-1)
        cv2.imshow("handcolor", handcolor)

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


