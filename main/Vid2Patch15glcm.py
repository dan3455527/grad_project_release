from full_frame_record import *
from glcm import *

if __name__ == "__main__":

  load_dir = "D:\\FullFrame\\Eric"
  save_dir = "D:\\FullFrame\\psize15_vid_glcm\\Eric"
  for i in range(0, 50):
    print(i)
    save_ls = []
    file_name = f"data_{i}.npy"
    vid = np.load(os.path.join(load_dir, file_name))
    for frame in vid:
      frame_info = Info.Frame(frame)
      face_data = Face(frame)
      rgb_image = frame_info.frame
      landmark52 = face_data.get_landmarks("52")
      face_data.get_segmented_face(image=rgb_image, patch_size=15)
      seg_face_ls = face_data.segmented_face
      data_process = Data(patch_size=15, data_length_limit=50, data=face_data.segmented_face)
      saved_data = glcm_full_list(seg_face_ls)
      save_ls.append(np.reshape(saved_data, (52, 15, 15, 3)))
    save_path = Utils().get_save_file_name(save_dir)
    Utils().mode_selected(save_ls, "save", save_path)

      




      