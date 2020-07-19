# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
# from lenet import LeNet   #syn
from DimResNN import DimResNN
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import keras

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="training_set",
                help="path to input dataset")
ap.add_argument("-t", "--target", type=str, default="target_set",
                help="path to input target")
ap.add_argument("-m", "--model", type=str, default="trained_model",
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

EPOCHS = 250
INIT_LR = 0.01
BS = 2

data = []
label = []

# grab the image paths and randomly shuffle them
print("[INFO] Loading Images...")
imagePaths_1 = sorted(list(paths.list_images(args["dataset"]))) # paths：这个用法是把dir路径下的所有图片名称变成一个列表
imagePaths_2 = sorted(list(paths.list_images(args["target"]))) # paths：这个用法是把dir路径下的所有图片名称变成一个列表
random.seed(42)
random.shuffle(imagePaths_1)
random.seed(42)
random.shuffle(imagePaths_2)

print(imagePaths_1)
print(imagePaths_2)

for imagePath in imagePaths_1:
    image_1 = cv2.imread(imagePath)
    image_1 = cv2.resize(image_1, (224, 224))
    image_1 = img_to_array(image_1)
    data.append(image_1)

for imagePath in imagePaths_2:
    image_2 = cv2.imread(imagePath)
    image_2 = cv2.resize(image_2, (224, 224))
    image_2 = img_to_array(image_2)  # img_to_array的作用是把整数转换成浮点数
    label.append(image_2)

data = np.array(data, dtype="float")
label = np.array(label, dtype="float")

(trainX, testX, trainY, testY) = train_test_split(data, label, test_size=0.25, random_state=42)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)


# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
#                          height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#                          horizontal_flip=True, fill_mode="nearest")

print("[INFO] Compiling Model...")
#model = LeNet.build(width=224, height=224, depth=3) #syn
model=DimResNN(depth=3,filters=3,kernal_size=3,use_bn=True) #filter=? syn
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss=keras.losses.mse, optimizer=opt,
              metrics=["accuracy"])               #syn  tensorflow2不支持optimizer

print("[INFO] Training Network...")
H = model.fit(trainX,
              trainY,
              batch_size=BS,
              epochs=EPOCHS,

              verbose=1,
              validation_data=(testX, testY),    #syn shape==(224,224,3)
             )
 
print("[INFO] Saving Model...")
model_base = args["model"] + '.h5'
model.save(model_base)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

print("[INFO] Completed...")