import cv2
import mediapipe as mp
import math
import time
import os
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import data, filters
from keras.models import load_model


def mode_selected(data, mode, path):
    if mode == "save":
        np.save(path, data)
    elif mode == "pred":
        data = data / 255
        model = load_model("./lstm_model_face.h5")
        pred = model.predict_classes(data)
        print(pred)


def standarize_data(data, data_length_limit=58):
    if len(data) > data_length_limit:
        counter = len(data)
        while counter > data_length_limit:
            if counter % 2 == 0:
                new_data = np.delete(data, 0, axis=0)
                data = new_data
            else:
                new_data = np.delete(data, -1, axis=0)
                data = new_data
            counter -= 1
        return new_data
    else:
        return data


def eye_blinking(coords):
    criteria = 0.13  # adjustable
    right_eyel = 133
    right_eyer = 33
    right_eyet = 159
    right_eyeb = 145
    left_eyel = 263
    left_eyer = 362
    left_eyet = 386
    left_eyeb = 374
    horizontal_right = math.dist(coords[right_eyel], coords[right_eyer])
    vertical_right = math.dist(coords[right_eyet], coords[right_eyeb])
    horizontal_left = math.dist(coords[left_eyel], coords[left_eyer])
    vertical_left = math.dist(coords[left_eyet], coords[left_eyeb])
    EAR_right = vertical_right / (2 * horizontal_right)
    EAR_left = vertical_left / (2 * horizontal_left)

    if EAR_right < criteria or EAR_left < criteria:
        return True
    else:
        return False


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
PTime = 0
CTime = 0
timer_time = 0
blink_count = 0
break_flag_EyeClose = False
# adjustable parameters below
data_length_limit = 28  # depends on the device performance
patch_size = 3  # size of each patch, has to be odd number
data_save_path = "/Users/huchengyou/Desktop/test"  # folder path

left_eye = [7, 33, 46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 133, 144, 145, 153, 154, 155, 157, 158,
            159, 160, 161, 163, 173, 246]
right_eye = [249, 263, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 362, 373, 374, 380, 381, 382,
             384, 385, 386, 387, 388, 390, 398, 466]
face_oval = [6, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 54, 56, 67, 68, 69, 71,
             103, 104, 108, 109, 110, 111, 112, 113, 114, 121, 122, 124, 127, 128, 130, 139, 143, 151,
             156, 162, 168, 174, 188, 189, 190, 193, 196, 197, 221, 222, 223, 224, 225, 226, 228, 229,
             230, 231, 232, 233, 243, 244, 245, 247, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
             261, 264, 265, 284, 286, 297, 298, 299, 301, 332, 333, 337, 338, 339, 341, 342, 343, 351,
             353, 356, 357, 359, 368, 372, 383, 389, 412, 413, 414, 417, 419, 441, 442, 443, 444, 445,
             446, 451, 452, 453, 463, 464, 465, 467]


def normalized_to_px(x_coord, y_coord, img_width, img_height):
    def _is_valid_coord(coord):
        return (coord > 0 or math.isclose(0, coord)) and (coord < 1 or math.isclose(1, coord))

    if not (_is_valid_coord(x_coord) and _is_valid_coord(y_coord)):
        return None
    x_px = min(math.floor(x_coord * img_width), img_width - 1)
    y_px = min(math.floor(y_coord * img_height), img_height - 1)
    return [x_px, y_px]


continuous_face_segment = []
timestamp = []
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    start = time.time()
    while cap.isOpened():
        success, image = cap.read()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_rows, image_cols, _ = image.shape
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                coords_point = []
                coords_norm = []
                for landmark in face_landmarks.landmark:
                    coords_point.append(normalized_to_px(
                        landmark.x, landmark.y, image_cols, image_rows))
                    coords_norm.append([landmark.x, landmark.y, landmark.z])
                left_coord = [coords_point[i] for i in left_eye]
                right_coord = [coords_point[i] for i in right_eye]
                face_coord = [coords_point[i] for i in face_oval]
                upper_face_coords = left_coord + right_coord + face_coord
                # patch segmentation of 173 point on face
                segmented_face = []
                for coord in upper_face_coords:
                    patch = image_gray[coord[1] - (patch_size // 2):coord[1] + (patch_size // 2 + 1),
                                       coord[0] - (patch_size // 2):coord[0] + (patch_size // 2 + 1)]
                    segmented_face.append(patch)

                continuous_face_segment.append(segmented_face)
                timestamp.append(time.time())
                if len(continuous_face_segment) < data_length_limit + 15:
                    cv2.putText(image, "not ready", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                elif len(continuous_face_segment) >= data_length_limit + 15:
                    cv2.putText(image, "ready to unlock", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                # show fps
                CTime = time.time()
                fps = int(1 / (CTime - PTime))
                cv2.putText(image, f'fps:{fps}', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                PTime = CTime

            end = time.time()
            i = float(format(end - start))
            j = i % 4
            x = len(continuous_face_segment)  # 資料筆數
            y = len(timestamp)  # 每筆資料的時間戳
            if x == (data_length_limit + 15):
                print("updating")
                del continuous_face_segment[0]
                del timestamp[0]
                data = np.array(continuous_face_segment)
                data_time_stamp = np.array(timestamp)
                pass
            if eye_blinking(coords_point):
                print('eye closed, start timer')
                cv2.putText(image, "eye blink", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                timer_time = time.time()
            if CTime - timer_time > 1 and timer_time != 0:
                print('get data')
                for i in range(len(data)):
                    time_stamp = data_time_stamp[i]
                    if time_stamp > (CTime - 2):
                        data_4s = data[i:]
                        print('save 4s data')
                        data_4s = standarize_data(
                            data_4s, data_length_limit=data_length_limit)
                        if data_4s.shape[0] == data_length_limit:
                            # TODO lbp image preprocessing here
                            data_4s_lbp = []
                            radius = patch_size // 2  # lbp範圍半徑取值
                            n_points = patch_size ** 2 - 1  # 領域像素點數
                            for idf, f in enumerate(data_4s):
                                for id, point in enumerate(f):
                                    lbp_img = local_binary_pattern(
                                        point, n_points, radius)
                                    edge_img = filters.sobel(point)
                                    data_4s_lbp.append(lbp_img)
                            """save data as .npy files"""
                            counter = 0
                            for files in os.listdir(data_save_path):
                                if "data_" in files:
                                    counter += 1
                            img_save_path = data_save_path + \
                                "/data_" + str(counter) + ".npy"
                            data_4s_lbp = np.array(data_4s_lbp)
                            data_4s_lbp = np.reshape(
                                data_4s_lbp, (data_length_limit, 173, 3, 3))
                            """recognition or pred"""
                            mode_selected(data_4s_lbp, "pred", img_save_path)

                        else:
                            # if this happens frequently, might needs to adjust data_length_limit
                            cv2.putText(image, "Redo Blinking", (700, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            break

                        break_flag_EyeClose = True
                        break
                timer_time = 0

            cv2.imshow('MediaPipe FaceMesh', image)

        if break_flag_EyeClose:
            break

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
