# Ignore the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# configure
style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# dl libraries
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixels
import cv2
from tqdm import tqdm
import os
from random import shuffle
from zipfile import ZipFile
from PIL import Image


# Data load
X = []
Z = []

IMG_SIZE = 150

FREE_DIR = "dataset/Free"
FULL_DIR = "dataset/Full"


def assign_label(img, label):
    return label


def make_train_data(label, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, label)

        path = os.path.join(DIR, img)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(np.array(img))
        Z.append(str(label))


# make 'Free' data
make_train_data("Free", FREE_DIR)

# make 'Full' data
make_train_data("Full", FULL_DIR)



# check some image
fig,ax = plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range(2):
        l = rn.randint(0, len(Z) - 1)
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])
os.makedirs("output_images", exist_ok=True)

plt.tight_layout()
plt.savefig("output_images/dataset_sample.png")
plt.show()

le = LabelEncoder()
Y = le.fit_transform(Z)
Y = to_categorical(Y,2)
X = np.array(X)
X = X/255
