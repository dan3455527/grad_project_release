from full_frame_record import *
from glcm import *

if __name__ == "__main__":

  load_dir = "D:\\FullFrame\\Eric"
  save_dir = "D:\\FullFrame\\psize15_vid_hog\\Eric"
  for i in range(0, 50):
    print(i)
    save_ls = []
    file_name = f"data_{i}.npy"
    vid = np.load(os.path.join(load_dir, file_name))
    for frame in vid:
      frame_info = Info.Frame(frame)
      face_data = Face(frame)
      gray_image = frame_info.gray_image
      landmark52 = face_data.get_landmarks("52")
      face_data.get_segmented_face(image=gray_image, patch_size=15)
      seg_face_ls = face_data.segmented_face
      data_process = Data(patch_size=15, data_length_limit=50, data=face_data.segmented_face)
      saved_data = np.reshape(seg_face_ls, (1, 52, 15, 15))
      # saved_data = data_process.convert_lbp(saved_data) # lbp convert
      # saved_data = data_process.convert_hog(saved_data, orientations=6, pixels_per_cell=(5, 5), cells_per_block=(3, 3)) # hog convert
      save_ls.append(np.reshape(saved_data, (52, 15, 15)))
    save_path = Utils().get_save_file_name(save_dir)
    Utils().mode_selected(save_ls, "save", save_path)

      




      