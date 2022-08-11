from full_frame_record import *

if __name__ == "__main__":
  continuous_data = []
  saved_data = []
  cap = cv2.VideoCapture(0)
  while cap.isOpened():
    success, image = cap.read()
    frame_info = Info.Frame(image)
    if not success:
      print("ingore emtpy frame")
      continue

    image = frame_info.frame
    gray_image = frame_info.gray_image
    image.flags.writeable = False

    face_data = Face(image)
    landmark52 = face_data.get_landmarks("52")
    face_data.get_segmented_face(gray_image=gray_image, patch_size=15)
    data_process = Data(patch_size=15, data_length_limit=50, data=face_data.segmented_face)
    window = Visulize(image)                                                                 
    if Utils().eye_blinking(face_data.coords_point, criteria=0.12):
      window.show_blink_promt()
      timer_time = time.time()
    window.show_fps()
    saved_data = data_process.record_data(continuous_data, image) 
    cv2.imshow("Mediapipe", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if break_flag_EyeClose:
      saved_data = data_process.convert_lbp(saved_data)
      pred = Utils().mode_selected(saved_data, 'pred', data_dim=(data_process.data_length_limit, 52, data_process.patch_size, data_process.patch_size), model_path="D:\\Documents\\graduation_project\\face_recognition\\face-recognition-via-texture\\lstmPsize15_lbp_3member.h5")
      print(pred)
      break
    if cv2.waitKey(5) &0xFF == 27:
      break
  
  cap.release()
  pass
