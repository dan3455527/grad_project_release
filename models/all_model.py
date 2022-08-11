from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, GRU
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam_v2
from keras.layers import BatchNormalization, Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report

class Model:
  def __init__(self, train_data, train_target, test_data, test_target) -> None:
    self.train_data = train_data
    self.train_target = train_target
    self.test_data = test_data
    self.test_target = test_target
    self.input_shape = train_data.shape[1:]
    pass

  def build_cnn(self):
    self.model = Sequential()
    self.model.add(Conv2D(32, (3, 3), input_shape=self.input_shape, activation="relu"))
    self.model.add(BatchNormalization())
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.2))
    self.model.add(Conv2D(32, (3, 3), activation="relu"))
    self.model.add(BatchNormalization())
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.2))
    self.model.add(Conv2D(64, (3, 3), activation="relu"))
    self.model.add(BatchNormalization())
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.2))
    self.model.add(Flatten())
    self.model.add(Dense(128, activation="relu"))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(self.train_target[1], activation="sigmoid"))
    self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    self.model.summary()
    return None

  def build_bilstm(self):
    self.model = Sequential()
    self.model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=self.input_shape))
    self.model.add(Bidirectional(LSTM(64)))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(64, activation="tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(32, activation="tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(16, activation = "tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(16, activation="tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(2, activation="softmax"))
    adam = adam_v2.Adam(learning_rate=0.0005)
    self.model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
    self.model.summary()
    return None
  
  def build_lstm(self):
    self.model = Sequential()
    self.model.add(LSTM(64, input_shape=self.input_shape))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(64, activation="tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(32, activation="tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(16, activation = "tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(16, activation="tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(3, activation="softmax"))
    adam = adam_v2.Adam(learning_rate=0.0006)
    self.model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
    self.model.summary()
    return None

  def build_gru(self):
    self.model = Sequential()
    self.model.add(GRU(64 ,input_shape=self.input_shape))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(64, activation="tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(32, activation="tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(16, activation = "tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(16, activation="tanh"))
    self.model.add(Dropout(0.1))
    self.model.add(Dense(3, activation="softmax"))
    adam = adam_v2.Adam(learning_rate=0.0005)
    self.model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
    self.model.summary()
    
  def fit_model(self, epoch=50, validation_split=0.2, batch_size=64):
    history = self.model.fit(self.train_data, self.train_target, epochs=epoch, validation_split=validation_split, batch_size=batch_size)

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model acc")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="lower right")
    plt.show()
    
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="lower right")
    plt.show()
    return None

class Data:
  def __init__(self, dir_path) -> None:
    self.dir_path = dir_path # input the list of all dir
    self.train_datasets = []
    self.test_datasets = []
    self.train_groud_truth = []
    self.test_ground_truth = []
    pass

  def create_datasets(self, data_shape=None):
    id = 0
    for dir in self.dir_path:
      for files in os.listdir(dir):
        data_arr = np.load(os.path.join(dir, files))
        if data_shape != None:
          data_arr = np.reshape(data_arr, data_shape)
        self.train_datasets.append(data_arr)
        self.train_groud_truth.append(id)
      id += 1

    # test split
    train_sets, test_sets, train_target, test_target = train_test_split(
      self.train_datasets, self.train_groud_truth, test_size=0.2, random_state=0)
    self.train_datasets = np.array(train_sets) / 255
    self.train_groud_truth = np.array(train_target)
    self.test_datasets = np.array(test_sets) /255
    self.test_ground_truth = np.array(test_target)

    # one hot
    self.train_groud_truth = to_categorical(train_target)
    self.test_ground_truth = to_categorical(test_target)

    return self.train_datasets, self.train_groud_truth, self.test_datasets, self.test_ground_truth



# if __name__ == "__main__":
#   data_dir = ["D:\\FullFrame\\psize15_lbp_perframe\\Daniel", "D:\\FullFrame\\psize15_lbp_preframe\\Eric"]
#   datasets = Data(data_dir)
#   datasets.create_datasets()
#   cnn = Model(datasets.train_datasets, \
#     datasets.train_groud_truth, datasets.test_datasets, datasets.test_ground_truth)
#   cnn.build_cnn()
#   cnn.fit_model(epoch=50, validation_split=0.2, batch_size=32)
#   cnn.model.save("./cnn_full_vid_BatchNormal.h5")
#   pass

# if __name__ == "__main__":
#   data_dir = ["D:\\FullFrame\\psize15_lbp_perframe\\Daniel", "D:\\FullFrame\\psize15_lbp_perframe\\Eric"]
#   datasets = Data(data_dir)
#   train_data, train_target, test_data, test_target = datasets.create_datasets((195, 60, 1))
#   infodoc = f"""train_data : {train_data.shape}\n
#     train_target: {train_target.shape}\n
#     test_data : {test_data.shape}\n
#     test_target : {test_target.shape}"""
#   print(infodoc)

#   cnn = Model(train_data, train_target, test_data, test_target)
#   cnn.build_cnn()
#   cnn.fit_model(epoch=50, validation_split=0.2, batch_size=32)
#   cnn.model.save("lbp_52_perframeCnn.h5")

#   model = load_model("lbp_52_perframeCnn.h5")
#   pred = model.predict(test_data)
#   predict_classes = np.argmax(np.round(pred), axis=1)
#   target_name = ["class{}".format(i) for i in range(2)]
#   print(classification_report(np.argmax(np.round(test_target), axis=1), predict_classes, target_names=target_name))

#   pass

if __name__ == "__main__":
  data_dir = ["D:/FullFrame/psize15_vid_glcm/Daniel", "D:/FullFrame/psize15_vid_glcm/Eric"]#, "D:/FullFrame/psize15_vid_hog/Derrick"]
  datasets = Data(data_dir)
  train_data, train_target, test_data, test_target = datasets.create_datasets((50, 35100))
  infodoc = f"""
  train_data : {train_data.shape}\n
  train_target: {train_target.shape}\n
  test_data : {test_data.shape}\n
  test_target : {test_target.shape}"""
  print(infodoc)

  lstm = Model(train_data, train_target, test_data, test_target)
  lstm.build_bilstm()
  lstm.fit_model(epoch=100, validation_split=0.2, batch_size=8)

  pred = lstm.model.predict(test_data)
  predict_classes = np.argmax(np.round(pred), axis=1)
  print(predict_classes)

  lstm.model.evaluate(test_data, test_target, verbose=1)
  lstm.model.save("bilstmPsize15_glcm_2member.h5")

  pass