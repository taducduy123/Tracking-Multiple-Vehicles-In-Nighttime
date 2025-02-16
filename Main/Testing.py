import numpy as np
import cv2
import time
import math
import HeadlightTrackingAndParing as htp
from pathlib import Path

# -------------------------------------------- URLs of Input Resources ---------------------------------------------------------
video_name = "CarsAtNight.mp4"
headlight_detection_model_name = "HeadlightDetector_16stg.xml"


script_dir = (Path(__file__).parent).parent             # Get the current script's directory
video_path = script_dir / "Resources" / "Video" / video_name
headlight_detection_model_path = script_dir / "Resources" / "Headlight Detection Model" / headlight_detection_model_name



# -------------------------------------------------- GUI SETTINGS ------------------------------------------------------
cap = cv2.VideoCapture(video_path)        # for similar videos: https://www.shutterstock.com/vi/video/search/similar/18908171

# Default Frames Per Second (FPS)
FPS = 1
frame_time = 1 / FPS  # Time per frame

# Scaling frame: how many times we want to scale a frame
X_SCALE = 1
Y_SCALE = 1

# Font
FONT = cv2.FONT_HERSHEY_SIMPLEX

# States of video
paused = False
hidden = False
pairingOn = True
trackingOn = True

# -------------------------------------------------- TRAINED MODEL -----------------------------------------------------
headlight_detector = cv2.CascadeClassifier(headlight_detection_model_path)




# --------------------------------------- PARAMETERS BEFOREHAND FOR ALGORITHM ------------------------------------------
max_num_objects = 30                                                 # number of concerned objects in each frame
dummy_headlight = htp.Headlight(0, 0, math.pi, math.pi/4)   # dummy headlight
at_most_five_consecutive_frames = []                                 # to record headlights in LAST 5 frames
frames = []                                                          # to record frames SO FAR: each frame contains headlights
association_matrices = []                                            # to record association matrices SO FAR: each matrix corresponds to a frame
pairing_indicators = []                                              # to record pairing indicators SO FAR: each paring indicator corresponds to a frame
paring_indicators_t_minus_1 = None                                   # to maintain the pairing indicator of the previous frame:




# ------------------------------------------------ LET'S START ---------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
def check_intersection_with_ROI(ROI, rect_pts):
    region_of_interest = np.array(ROI, dtype=np.int32)
    rectangle = np.array(rect_pts, dtype=np.int32)

    # Check intersection between Region Of Interest(ROI) and a detected headlight
    intersection_area, intersection_pts = cv2.intersectConvexConvex(region_of_interest, rectangle)

    if intersection_area < 0.0000000000001:
        return False
    return True


def get_partner_index(current_index, pairing_indicator):
    partner_index = -1
    for pair in pairing_indicator:
        if pair[0] == current_index:
            partner_index = pair[1]
            break
        if pair[1] == current_index:
            partner_index = pair[0]
            break

    return partner_index


def check_two_headlights_same_position(position1, position2):
    if position1[0] == position2[0] and position1[1] == position2[1]:
        return True
    return False


# ----------------------------------------------------------------------------------------------------------------------
association_matrix_marker = 0
pairing_indicator_marker = 0
frame_counter = 0

while cap.isOpened():
    if not paused:
        success, frame = cap.read()

        if success:
            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)

            start_time = time.perf_counter()        # return current time in SECOND


            # Convert frame into gray frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Applying Model to Detect Single Headlight
            results = headlight_detector.detectMultiScale(gray, 1.3, 2, 75)

            # Get frame's dimension
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            # Region Of Interest (ROI)
            pt1 = (int(0), int(2/5 * frame_height))
            pt2 = (int(3/10 * frame_width), int(2/5 * frame_height))
            pt3 = (int(frame_width), int(9/10 * frame_height))
            pt4 = (int(frame_width), int(frame_height))
            pt5 = (int(0), int(frame_height))
            ROI = [pt1, pt2, pt3, pt4, pt5]

            # Draw ROI
            cv2.polylines(frame, [np.array(ROI, dtype=np.int32)], True, (0, 0, 255), 2)

            # Collect detected headlights
            headlights = np.empty(max_num_objects, dtype= htp.Headlight)
            association_matrix_index = 0
            for (x, y, w, h) in results:
                if check_intersection_with_ROI(ROI, [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]):
                    # position of a headlight
                    x_centroid = x + (w / 2)
                    y_centroid = y + (h / 2)

                    # area of a headlight
                    area = (math.pi / 4) * (w**2)
                    if h < w:
                        area = (math.pi / 4) * (h**2)

                    # shape of a headlight
                    enclosing_box_area = w * h
                    shape = area / enclosing_box_area

                    # Initialize a headlight
                    headlights[association_matrix_index] = htp.Headlight(x_centroid, y_centroid, area, shape)
                    association_matrix_index = association_matrix_index + 1

            # Add dummy headlights
            while association_matrix_index < max_num_objects:
                headlights[association_matrix_index] = dummy_headlight
                association_matrix_index = association_matrix_index + 1

            # Maintain last 5 consecutive frames
            if len(at_most_five_consecutive_frames) >= 5:
                at_most_five_consecutive_frames.pop(0)
            at_most_five_consecutive_frames.append(headlights)
            frames.append(headlights)

            # Apply Tracking and Paring Algorithm
            if len(at_most_five_consecutive_frames) > 1:
                association_matrix, pairing_indicator = htp.headlight_tracking_and_pairing(at_most_five_consecutive_frames, association_matrices, paring_indicators_t_minus_1)

                # Accumulate result after running the algorithm
                association_matrices.append(association_matrix)
                pairing_indicators.append(pairing_indicator)
                paring_indicators_t_minus_1 = pairing_indicator


            # Draw tracking
            if trackingOn:
                for association_matrix_index in range(association_matrix_marker, len(association_matrices), 1):
                    headlight_index = 0
                    for headlight in frames[association_matrix_index]:
                        headlight_t_minus_1_index = headlight_index
                        headlight_t_index = htp.get_forward_headlight_index(headlight_t_minus_1_index, association_matrices[association_matrix_index])
                        headlight_t = frames[association_matrix_index + 1][headlight_t_index]

                        headlight_t_minus_1_position = (int(headlight.x),    int(headlight.y))
                        headlight_t_position         = (int(headlight_t.x),  int(headlight_t.y))

                        # Draw tracking line between 2 headlights only if all following conditions are held
                        # - These 2 headlights are neither dummy headlights
                        # - The distance between these < threshold
                        # - Every headlight of these must have proper partner (position of proper partner != position of headlight)
                        if association_matrix_index == 0:               # 2 first frames
                            if ((headlight_t_minus_1_position[0] != dummy_headlight.x and headlight_t_minus_1_position[1] != dummy_headlight.y and headlight_t_position[0] != dummy_headlight.x and headlight_t_position[1] != dummy_headlight.y)
                                and (abs(headlight_t_minus_1_position[0] - headlight_t_position[0]) < 50 and abs(headlight_t_minus_1_position[1] - headlight_t_position[1]) < 60)):
                                cv2.line(frame, headlight_t_minus_1_position, headlight_t_position, (255, 0, 0), 3)
                                # print(f"{headlight_t_minus_1_position} ------------------- {headlight_t_position}")
                        else:
                            headlight_t_minus_1_partner_index = get_partner_index(headlight_t_minus_1_index, pairing_indicators[association_matrix_index - 1])
                            headlight_t_partner_index = get_partner_index(headlight_t_index, pairing_indicators[association_matrix_index])

                            headlight_t_minus_1_partner = frames[association_matrix_index][headlight_t_minus_1_partner_index]
                            headlight_t_partner = frames[association_matrix_index + 1][headlight_t_partner_index]

                            headlight_t_minus_1_partner_position = (int(headlight_t_minus_1_partner.x), int(headlight_t_minus_1_partner.y))
                            headlight_t_partner_position = (int(headlight_t_partner.x), int(headlight_t_partner.y))

                            if ((headlight_t_minus_1_position[0] != dummy_headlight.x and headlight_t_minus_1_position[1] != dummy_headlight.y and headlight_t_position[0] != dummy_headlight.x and headlight_t_position[1] != dummy_headlight.y)
                                and (not check_two_headlights_same_position(headlight_t_minus_1_position, headlight_t_minus_1_partner_position))
                                and (not check_two_headlights_same_position(headlight_t_position, headlight_t_partner_position))
                                and (abs(headlight_t_minus_1_position[0] - headlight_t_position[0]) < 50 and abs(headlight_t_minus_1_position[1] - headlight_t_position[1]) < 60)):
                                cv2.line(frame, headlight_t_minus_1_position, headlight_t_position, (255, 0, 0), 3)
                                # print(f"{headlight_t_minus_1_position} ------------------- {headlight_t_position}")

                        headlight_index = headlight_index + 1
                    # print("***********************************************************************************************************************"
                    association_matrix_index = association_matrix_index + 1


            # Draw pairing
            if pairingOn:
                for indicator_index in range(pairing_indicator_marker, len(pairing_indicators), 1):
                    for pair in pairing_indicators[indicator_index]:
                        headlight = frames[indicator_index + 1][pair[0]]
                        partner   = frames[indicator_index + 1][pair[1]]

                        headlight_position = (int(headlight.x), int(headlight.y))
                        partner_position = (int(partner.x), int(partner.y))

                        # Draw pairing line between 2 headlights only if all following conditions are held
                        # - These 2 headlights are neither dummy headlights
                        if ((headlight_position[0] != dummy_headlight.x and headlight_position[1] != dummy_headlight.y and partner_position[0] != dummy_headlight.x and partner_position[1] != dummy_headlight.y)):
                            cv2.line(frame, headlight_position, partner_position, (128, 0, 128), 2)
                            #print(f"{headlight_position} ------------------- {partner_position}")
                    #print("***********************************************************************************************************************")
                    indicator_index = indicator_index + 1


            # Drawing rectangle bounding each headlight with Annotation
            for (x, y, w, h) in results:
                if check_intersection_with_ROI(ROI, [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]):
                    rec = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if not hidden:
                        cv2.putText(rec, f"{int(x + (w/2)), int(y + (h/2))}", (x-30, y-5), FONT, 0.6, (0, 255, 0), 2)

            # Show the frame and related information
            x = 10
            y = 50
            cv2.putText(frame, f"Frame Counter: {frame_counter}", (x, y), FONT, 1.5, (255, 255, 255), 4)
            cv2.putText(frame, f"Current FPS: {FPS}", (x, y + 70), FONT, 1.5, (255, 255, 255), 4)
            cv2.putText(frame, f"--------: Tracking Line", (x, y + 140), FONT, 1.5, (255, 0, 0), 4)
            cv2.putText(frame, f"--------: Paring Line", (x, y + 210), FONT, 1.5, (128, 0, 128), 4)
            frame = cv2.resize(frame, (0, 0), fx = X_SCALE, fy = Y_SCALE)       # Make frame to fit the screen
            cv2.imshow("Multiple Headlights Tracking And Paring: [P] to PAUSE/CONTINUE. [Q] to STOP. [C] to CLEAR lines. [H] to HIDE/UNHIDE coordinates. [A] to show/unshow TRACKING. [S] to show/unshow PAIRING. [+]/[-] to SPEED/SLOW.", frame)

            # Count number of frames passed
            frame_counter = frame_counter + 1

            # Calculate processing time for this frame
            processing_time = time.perf_counter() - start_time
            remaining_time = frame_time - processing_time
            print(f"Max Performance: {int(1/processing_time)} FPS")
            if FPS > int(1/processing_time):
                FPS = int(1/processing_time)

            # Sleep during the remaining time
            if remaining_time > 0:
                time.sleep(remaining_time)


            # Break loop if 'q' is pressed
            key = cv2.waitKey(1)
            if key == ord("q") or key == ord("Q"):
                break
            # Pause program if 'p' is pressed
            if key == ord("p") or key == ord("P"):
                paused = True
            # Hide/Unhide coordinates if 'h' is pressed
            if key == ord("h") or key == ord("H"):
                if not hidden:
                    hidden = True
                else:
                    hidden = False
            # Show/Unshow tracking lines if 'a' is pressed
            if key == ord("a") or key == ord("A"):
                if trackingOn:
                    trackingOn = False
                else:
                    trackingOn = True
            # Show/Unshow pairing lines if 'a' is pressed
            if key == ord("s") or key == ord("S"):
                if pairingOn:
                    pairingOn = False
                else:
                    pairingOn = True
            # Clear drawing lines so far if 'c' is pressed
            if key == ord("c") or key == ord("C"):
                association_matrix_marker = len(association_matrices) - 1
                pairing_indicator_marker = len(pairing_indicators) - 1
            # Speed Up FPS when [+] is pressed
            if key == 61:
                if FPS < int(1/processing_time):
                    FPS = FPS + 1
                    frame_time = 1 / FPS
            # Slow Down FPS when [-] is pressed
            if key == 45:
                if FPS > 1:
                    FPS = FPS - 1
                    frame_time = 1 / FPS
        # If reached end of video
        else:
            break
    else:
        # Break loop if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord("q") or key == ord("Q"):
            break
        # Continue program if 'p' is pressed again
        if key == ord("p") or key == ord("P"):
            paused = False





cap.release()
cv2.destroyAllWindows()