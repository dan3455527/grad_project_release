"""all function for graduate project"""
import mediapipe as mp
import cv2
import numpy as np
import time
import math
import os
from skimage import feature as ft
from skimage.feature import local_binary_pattern
from keras.models import load_model

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
  max_num_faces = 1,
  refine_landmarks = True,
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5
)
Ptime = Ctime = timer_time = blink_count = 0
break_flag_EyeClose = False
timestamp = []
class Info:
  """Information of all relative constance
  """
  class LandmarkID:
    """land mark id for each face portion"""
    def __init__(self) -> None:
      self.left_eye = [7, 33, 46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 133, 144, 145, 153, 154, 155, 157, 158,
            159, 160, 161, 163, 173, 246]
      self.right_eye = [249, 263, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 362, 373, 374, 380, 381, 382,
            384, 385, 386, 387, 388, 390, 398, 466]
      self.face_oval = [6, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 54, 56, 67, 68, 69, 71,
            103, 104, 108, 109, 110, 111, 112, 113, 114, 121, 122, 124, 127, 128, 130, 139, 143, 151,
            156, 162, 168, 174, 188, 189, 190, 193, 196, 197, 221, 222, 223, 224, 225, 226, 228, 229,
            230, 231, 232, 233, 243, 244, 245, 247, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
            261, 264, 265, 284, 286, 297, 298, 299, 301, 332, 333, 337, 338, 339, 341, 342, 343, 351,
            353, 356, 357, 359, 368, 372, 383, 389, 412, 413, 414, 417, 419, 441, 442, 443, 444, 445,
            446, 451, 452, 453, 463, 464, 465, 467]
      pass
  
  class Frame:
    """Frame info"""
    def __init__(self, frame) -> None:
      """
      Args:
      ---
        frame: frame image in BGR format
      Return:
      ---
        gray_image: grayscale image of frame
        frame: frame image in RGB format
      """
      self.gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      self.frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
      pass

class Utils:
  """utilize function"""
  def normalized_to_px(x_coord, y_coord, img_width, img_height) -> list:
    """turn normalized coordinate back to pixel coordinate

    Args:
    ---
      x_coord: normalize x coordinate
      y_coord: normalize y coordinate
      img_width: frame width or full image width
      img_height: frame height or full image height

    Return:
    ---
      [x_px, y_px]: list of 2 pixel coordinate
    """
    def _is_valid_coord(coord):
      return (coord > 0 or math.isclose(0, coord)) and (coord < 1 or math.isclose(1, coord))

    if not (_is_valid_coord(x_coord) and _is_valid_coord(y_coord)):
      return None
    x_px = min(math.floor(x_coord * img_width), img_width - 1)
    y_px = min(math.floor(y_coord * img_height), img_height - 1)
    return [x_px, y_px]

  def eye_blinking(self, coords, criteria=0.12) -> bool:
    """eye blinking detector and timer starter
    
    Args:
    ---
      coords: list of face coordinate point in pixel
      criteria: eye aspect ratio tracking the blink
    
    Return:
    ---
      Booling value of blinking status
    """
    global timer_time
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
      timer_time = time.time()
      return True
    else:
      return False

  def get_save_file_name(self, save_path) -> str:
    """get the started file name and id from running the directory
    
    Args:
    ---
      save_path: directory to save the data
    
    Return:
    ---
      img_save_path: direct path to save the data in .npy
    """
    counter = 0
    for files in os.listdir(save_path):
      if "data_" in files:
        counter += 1
    img_save_path = save_path + "/data_" + str(counter) + ".npy"
    return img_save_path
  
  def mode_selected(self, data, mode, save_path=None, data_dim=None, model_path=None):
    """mode selection to run saving progess or prediction
    
    Args:
    ---
      data: the list of data ready to save
      mode: "save" or "pred"
      save_path: path to save file
      data_dim: dimension of the data to be reshape for the model
      model_path: path of the pre-trained model

    Return:
    ---
      pred: prediction of the data
      pred_class: prediction of the data in classes
    """
    data = np.array(data)
    if mode == "save":
      np.save(save_path, data)
    elif mode == "pred":
      data = np.reshape(data, (1, data_dim[0], data_dim[1]*data_dim[2]*data_dim[3]))
      data = data / 255
      model = load_model(model_path)
      pred = model.predict(data)
      pred_class = np.argmax(pred, axis=1)
      return pred, pred_class


class Face:
  def __init__(self, frame) -> None:
    """initialize face information for each frame or image

    Args:
    ---
    frame: input frame or image
    """
    global face_mesh
    self.frame = frame
    self.results = face_mesh.process(frame)
    self.segmented_face = []
    pass

  def get_landmarks(self, mode) -> list:
    """get the list of coordinate of each point on the face

    Args:
    ---
      mode : 52 or 173
    
    Return:
    ---
      face_coord: all coordinate of face in pixel, also saved as an Attribute of this class
    
    Raises:
    ---
      ValueError: if the mode is not allowed number
    """
    if self.results.multi_face_landmarks:
      for face_landmarks in self.results.multi_face_landmarks:
        self.coords_point = []
        self.coords_norm = []
        for landmark in face_landmarks.landmark:
          self.coords_point.append(Utils.normalized_to_px(landmark.x, landmark.y, self.frame.shape[1], self.frame.shape[0]))
          self.coords_norm.append([landmark.x, landmark.y, landmark.z])
        left_coord = [self.coords_point[i] for i in Info.LandmarkID().left_eye]
        right_coord = [self.coords_point[i] for i in Info.LandmarkID().right_eye]
        face_coord = [self.coords_point[i] for i in Info.LandmarkID().face_oval]
        if mode == "52":
          self.face_coord = left_coord + right_coord
          return self.face_coord
        elif mode == "173":
          self.face_coord = left_coord + right_coord + face_coord
          return self.face_coord
        else:
          raise ValueError("mode must be 52 or 173")

  def get_segmented_face(self, image, patch_size) -> list:
    """get the list of segmented face by the patch size input each frame
    
    Args:
    ---
      image: gray scale image input
      patch_size: size of segmentation (must be odd integer)

    Return:
    ---
      segmented_face: list of segmented face

    """
    for _coord in self.face_coord:
      patch = image[_coord[1] - (patch_size // 2):_coord[1] + (patch_size // 2 + 1) , \
        _coord[0] - (patch_size // 2):_coord[0] + (patch_size // 2 + 1)]
      self.segmented_face.append(patch)
    return self.segmented_face
class Data:
  """data related function"""
  def __init__(self, patch_size, data_length_limit, data) -> None:
    """initialize data related variable
    
    Args:
    ---
      patch_size: patch size of the image
      data_length_limit: maximum saving frame amount
      data: list of data by frame
    """
    self.patch_size = patch_size
    self.data_length_limit = data_length_limit
    self.data = data
    self.counter = 0
    pass

  def record_data(self, stacked_data, frame, save_time_length=2):
    """recording the temporal data by setting time length, if time exceed the setted data_lenght limit
    will started to delete the very first data
    
    Args:
    ---
      stacked_data: list to constantly save the data of each frame
      frame: frame or image in RGB format
      save_time_length: time length to save the file
    
    Return:
    ---
      stacked_data: updated the saving data
    """
    global Ctime, Ptime, timer_time, break_flag_EyeClose, timestamp
    Ctime = time.time()
    stacked_data.append(self.data)
    timestamp.append(time.time())
    Visulize(frame).show_record_status(stacked_data, self.data_length_limit)
    data_length = len(stacked_data)
    if data_length == (self.data_length_limit + 15):
      print("updating")
      del stacked_data[0]
      del timestamp[0]
      self.data = np.array(stacked_data)
      data_time_stamp = np.array(timestamp)
      if Ctime - timer_time > (save_time_length/2) and timer_time != 0:
        print("get data")
        for _i in range(len(self.data)):
          time_stamp = data_time_stamp[_i]
          if time_stamp > (Ctime - save_time_length):
            stacked_data = self.data[_i:] 
            print(f"{save_time_length}s data saved")
            stacked_data = self.standarize_data(stacked_data)
            if stacked_data.shape[0] < self.data_length_limit:
              # TODO show redo promt
              break
            break_flag_EyeClose = True
            break
        timer_time = 0
    return stacked_data

  def standarize_data(self, data) -> np.ndarray:
    """standarize the data to setted length limit if length exceed

    Args:
    ---
      data: the temporal data to standarize the length
    
    Return:
    ---
      data: update the fixed data

    """
    if len(data) > self.data_length_limit:
      self.counter = len(data)
      while self.counter > self.data_length_limit:
        if self.counter % 2 == 0:
          new_data = np.delete(data, 0, axis=0)
          data = new_data
        else:
          new_data = np.delete(data, -1, axis=0)
          data = new_data
        self.counter -= 1
      return data
    else:
      return data

  def convert_lbp(self, data) -> list:
    """convert the input data into lbp

    Args:
    ---
      data: temporal data ready for convert

    Return:
    ---
      saved_data_lbp: data converted, also saved as an Attribute of the class
    """
    self.saved_data_lbp = []
    radius = self.patch_size // 2
    n_points = 8 * radius
    for _idf, _f in enumerate(data):
      for _id, _point in enumerate(_f):
        lbp_img = local_binary_pattern(_point, n_points, radius)
        lbp_img = (lbp_img / np.amax(lbp_img)) * 255
        lbp_img = lbp_img.astype(np.uint8)
        self.saved_data_lbp.append(lbp_img)
    return self.saved_data_lbp

  def convert_hog(self, data, orientations, pixels_per_cell, cells_per_block):
    """convert the input data into hog

    Args:
    ---
      data: temporal data ready for convert
      orientatoins:
      pixels_per_cell:
      cells_per_block:
    Return:
    ---
      saved_data_hog: data converted
    
    """
    saved_data_hog = []
    for _idf, _f in enumerate(data):
      for _id, _point in enumerate(_f):
        features = ft.hog(_point, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)
        saved_data_hog.append(features[1])
    return saved_data_hog

  def convert_glcm(self, data):
    pass
    

class Visulize:
  """visulaization on the output window"""
  def __init__(self, frame) -> None:
    """
    Args:
    ---
      frame : output frame to be shown on the window
    """
    self.frame = frame
    pass

  def show_fps(self):
    """show the current fps status"""
    global Ctime, Ptime
    Ctime = time.time()
    fps = int(1/ (Ctime - Ptime))
    cv2.putText(self.frame, f"FPS:{fps}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    Ptime = Ctime
    pass

  def show_blink_promt(self):
    """show the blink prompt"""
    print("eye closed, start timer")
    cv2.putText(self.frame, "eye blink", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    pass
  
  def show_record_status(self, continuous_data, data_length_limit):
    """show the record status, Not ready and ready"""
    if len(continuous_data) < data_length_limit + 15:
      cv2.putText(self.frame, "Not Ready", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    elif len(continuous_data) >= data_length_limit + 15:
      cv2.putText(self.frame, "Ready", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 0, 255), 2)
    pass

# main code for recording and prediction below
# if __name__ == "__main__":
#   continuous_data = []
#   saved_data = []
#   cap = cv2.VideoCapture(0)
#   while cap.isOpened():
#     success, image = cap.read()
#     frame_info = Info.Frame(image)
#     if not success:
#       print("ingore emtpy frame")
#       continue

#     image = frame_info.frame
#     gray_image = frame_info.gray_image
#     image.flags.writeable = False

#     face_data = Face(image)
#     landmark52 = face_data.get_landmarks("52")
#     face_data.get_segmented_face(gray_image=gray_image, patch_size=15)
#     data_process = Data(patch_size=15, data_length_limit=50, data=face_data.segmented_face)
#     window = Visulize(image)                                                                 
#     if Utils().eye_blinking(face_data.coords_point, criteria=0.12):
#       window.show_blink_promt()
#       timer_time = time.time()
#     window.show_fps()
#     saved_data = data_process.record_data(continuous_data, image) 
#     cv2.imshow("Mediapipe", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

#     if break_flag_EyeClose:
#       saved_data = data_process.convert_lbp(saved_data)
#       saved_data = np.reshape(saved_data, (50, 52, 15, 15))
#       pred = Utils().mode_selected(saved_data, 'pred', data_dim=(data_process.data_length_limit, 52, data_process.patch_size, data_process.patch_size), model_path="D:\\Documents\\graduation_project\\face_recognition\\face-recognition-via-texture\\lstmPsize15_lbp_3member.h5")
#       print(pred)
#       break
#     if cv2.waitKey(5) &0xFF == 27:
#       break
  
#   cap.release()
#   pass

""" main code for reformate frame data"""
# if __name__ == "__main__":
#   """create image datasets"""
#   load_dir = "D:\\FullFrame\\Daniel"
#   for _files in os.listdir(load_dir):
#     vid = np.load(os.path.join(load_dir, _files))
#     frame_length = np.shape(vid)[0]
#     for _i in range(frame_length):
#       img = im.fromarray(vid[_i])
#       img = img.resize((80, 60))
#       save_path = Utils().get_save_file_name("D:\\FullFrame\\img_datasets\\Daniel")
#       np.save(save_path, img)
#       print(f"img {_i} saved")
#   pass

# if __name__ == "__main__":
#   load_dir = "D:\\FullFrame\\Daniel"
#   for _files in os.listdir(load_dir):
#     vid = np.load(os.path.join(load_dir, _files))
#     frame_length = np.shape(vid)[0]
#     for _i in range(frame_length):
#       img = vid[_i]
#       gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#       face_data = Face(img)
#       landmark52 = face_data.get_landmarks("52")
#       face_data.get_segmented_face(gray_image=gray_img, patch_size=5)
#       seg_face_ls = face_data.segmented_face
#       D = Data(patch_size=5, data_length_limit=1, data=seg_face_ls)
#       saved_data = D.convert_lbp(seg_face_ls)
#       save_path = Utils().get_save_file_name("D:\\FullFrame\\frame_image_png\\Daniel")
#       Utils().mode_selected(seg_face_ls, "save", save_path)

# if __name__ == "__main__":
#   load_dir = "D:/FullFrame/Daniel"
#   save_dir = "D:/FullFrame/psize15_vid/Daniel"
#   for file in os.listdir(save_dir):
#     save_ls = []
#     vid = np.load(os.path.join(save_dir, file))
#     for frame in vid:
#       frame_info = Info.Frame(frame)
#       face_data = Face(frame)
#       gray_image = frame_info.gray_image
#       landmark52 = face_data.get_landmarks("52")
#       face_data.get_segmented_face(gray_image=gray_image)
#       seg_face_ls = face_data.segmented_face
#       data_process = Data(patch_size=15, data_length_limit=50, data=face_data.segmented_face)
#       saved_data = np.array(data_process.convert_lbp(seg_face_ls))
#       save_ls.append(saved_data)
#     save_path = Utils().get_save_file_name(save_dir)
#     Utils().mode_selected(save_ls, "save", save_path)