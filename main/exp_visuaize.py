import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
import numpy as np
import os

model = load_model("/Volumes/Samsung T7/Face_datasets/trained_model/gruPsize15_lbp_3member.h5")
model_static = load_model("/Volumes/Samsung T7/Face_datasets/trained_model/static_lbp.h5")

static_dir = [ #input shape = (195, 60)
  "/Volumes/Samsung T7/Face_datasets/static_lbp/dan_noise_30",  
  "/Volumes/Samsung T7/Face_datasets/static_lbp/eric_noise_30",
  "/Volumes/Samsung T7/Face_datasets/static_lbp/der_noise_30",  
  "/Volumes/Samsung T7/Face_datasets/static_lbp/der_jr_30",  
  "/Volumes/Samsung T7/Face_datasets/static_lbp/sim_30",  
  "/Volumes/Samsung T7/Face_datasets/static_lbp/zheng_30"  
]
dynamic_dir = [ # input shape = (50, 11700)
  "/Volumes/Samsung T7/Face_datasets/psize15_vid_lbp/Daniel_noise", #30
  "/Volumes/Samsung T7/Face_datasets/psize15_vid_lbp/Eric_noise",#20
  "/Volumes/Samsung T7/Face_datasets/psize15_vid_lbp/Derrick_noise",#30
  "/Volumes/Samsung T7/Face_datasets/psize15_vid_lbp/DerrickJr",#20 4350-4599
  "/Volumes/Samsung T7/Face_datasets/psize15_vid_lbp/Simon",#20 4600-4849
  "/Volumes/Samsung T7/Face_datasets/psize15_vid_lbp/Zheng"#20 4850-5099
]

class_dict = {
  0:"Daniel",
  1:"Eric",
  2:"Derrick",
  3:"DerrickJr",
  4:"Simon",  
  5:"Zheng"
}
confidence_thred = 0.95
dir = static_dir

test_datasets = []
test_target = []
for class_id in range(len(dir)):
  for root, dirs, files in os.walk(dir[class_id]):
    for file in files:
      arr = np.load(os.path.join(root, file))
      arr = np.reshape(arr, (195, 60)) /255
      test_datasets.append(arr)
      test_target.append(class_id)

test_datasets = np.array(test_datasets)
test_target = np.array(test_target)

pred = model_static.predict(test_datasets)
all_confidence = []
false_counter = 0
unknown_detect_count = 0
over_thred_count = 0
unknow_id_ls = []
tp_count = 0 # number of known and pred known  (True Positive)
tn_count = 0 # number of known but pred unknown (True Negative)
fp_count = 0 # number of unknown but pred known (False Positive)
fn_count = 0 # number of unknown and pred unknown (False Negative)

for idx, prediction in enumerate(pred):
  if class_dict[np.argmax(prediction)]==class_dict[test_target[idx]]: #knowned and correct predictionconfidence
    all_confidence.append(np.max(prediction))
    over_thred_count += 1
  # confusion matrix count:
  if test_target[idx] < 3:
    if np.max(prediction) > confidence_thred:
      tp_count += 1
    else:
      fn_count += 1
  elif test_target[idx] >= 3:
    if np.max(prediction) > confidence_thred:
      fp_count += 1
    else:
      tn_count += 1
      unknow_id_ls.append(idx)
  print(
    f"No.{idx:03d}--prediction : {prediction}",
    f"label: {class_dict[test_target[idx]]:10s}",
    f"confidence: {np.max(prediction):.2f}", 
    f"right class:{str((np.max(prediction) > confidence_thred) and (class_dict[np.argmax(prediction)]==class_dict[test_target[idx]])):5s}" # right prediction and over 85% threshold
    )
  if class_dict[np.argmax(prediction)]!=class_dict[test_target[idx]]: #wrong prediction, whether known or unknown
    false_counter += 1
  if (test_target[idx] >= 3) and (np.max(prediction) < confidence_thred):
    unknown_detect_count += 1
avg_confidence = np.sum(np.array(all_confidence)) * 100 / over_thred_count

print(f"average confidence for = {avg_confidence:.2f}%")
print(f"False ratio : {false_counter/ len(pred) * 100:.1f}%, accuracy :{100.0 - (false_counter/ len(pred) * 100):.1f}%")
print(f"unknown detection: {unknown_detect_count}")
# print confusion matrix

print(
  "{:<10s}{:<10s}{:<10s}\n".format("", "Positive", "Negative"),
  "{:<10s}{:<10d}{:<10d}\n".format("True", tp_count, fn_count),
  "{:<10s}{:<10d}{:<10d}\n".format("False", fp_count, tn_count)
)
# get statistic
sensitivity = tp_count / (tp_count + fn_count)
specificity = tn_count / (tn_count + fp_count)
precision = tp_count / (tp_count + fp_count)
accuracy = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)
f1_score = 2 / (1/ precision + 1/ sensitivity)
print(
  "known / unknown statistic\n"
  "{:<15s}{:6.4f}\n".format("accuracy", accuracy),
  "{:<15s}{:6.4f}\n".format("precision", precision),
  "{:<15s}{:6.4f}\n".format("sensitivity", sensitivity),
  "{:<15s}{:6.4f}\n".format("specificity", specificity),
  "{:<15s}{:6.4f}\n".format("f1 score", f1_score)
)
print(unknow_id_ls, len(unknow_id_ls))