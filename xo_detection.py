import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
# Initialize lists for cross and zeroes images
train_images_x = []
train_images_o = []

# Initialize labels array
train_labels = np.array([1] * 25 + [0] * 25)

# Load cross images
for i in range(181, 206):
    img1 = cv2.imread('./icg-freshers-data-science-competition/Dataset/Train/Cross/' + str(i) + '.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (28, 28), interpolation=cv2.INTER_AREA)
    train_images_x.append(img1)

# Load zeroes images
for i in range(594, 619):
    img1 = cv2.imread('./icg-freshers-data-science-competition/Dataset/Train/Zeroes/' + str(i) + '.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (28, 28), interpolation=cv2.INTER_AREA)
    train_images_o.append(img1)

# Convert lists to numpy arrays
train_images_x = np.array(train_images_x)
train_images_o = np.array(train_images_o)

# Concatenate cross and zeroes images
train_images = np.concatenate((train_images_x, train_images_o))

# Augment cross images
for i in range(181, 206):
    img1 = cv2.imread('./icg-freshers-data-science-competition/Dataset/Train/Cross/' + str(i) + '.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (28, 28), interpolation=cv2.INTER_AREA)
    
    img = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    train_images = np.concatenate((train_images, img[np.newaxis, :, :]))
    train_labels = np.append(train_labels, 1)

    img = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    train_images = np.concatenate((train_images, img[np.newaxis, :, :]))
    train_labels = np.append(train_labels, 1)

    img = cv2.rotate(img1, cv2.ROTATE_180)
    train_images = np.concatenate((train_images, img[np.newaxis, :, :]))
    train_labels = np.append(train_labels, 1)

# Augment zeroes images
for i in range(594, 619):
    img1 = cv2.imread('./icg-freshers-data-science-competition/Dataset/Train/Zeroes/' + str(i) + '.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (28, 28), interpolation=cv2.INTER_AREA)

    img = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    train_images = np.concatenate((train_images, img[np.newaxis, :, :]))
    train_labels = np.append(train_labels, 0)

    img = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    train_images = np.concatenate((train_images, img[np.newaxis, :, :]))
    train_labels = np.append(train_labels, 0)

    img = cv2.rotate(img1, cv2.ROTATE_180)
    train_images = np.concatenate((train_images, img[np.newaxis, :, :]))
    train_labels = np.append(train_labels, 0)



# Load labels from CSV
df = pd.read_csv('./icg-freshers-data-science-competition/Dataset/Train/Grid_labels.csv')

# Extract tiles and labels from CSV
for i in range(0, 47):
    row = df.iloc[i]
    img_path = './icg-freshers-data-science-competition/Dataset/Train/Grids/' + str(row['ID']) + '.png'
    im = cv2.imread("icg-freshers-data-science-competition/Dataset/Train/Grids/"+str(row['ID']) +".png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    tiles = []
    tiles.append(cv2.resize(im[59:176, 144:261].copy(), (28, 28), interpolation=cv2.INTER_AREA))
    tiles.append(cv2.resize(im[59:176, 269:382].copy(), (28, 28), interpolation=cv2.INTER_AREA))
    tiles.append(cv2.resize(im[59:176, 389:512].copy(), (28, 28), interpolation=cv2.INTER_AREA))
    tiles.append(cv2.resize(im[183:296, 144:261].copy(), (28, 28), interpolation=cv2.INTER_AREA))
    tiles.append(cv2.resize(im[183:296, 269:382].copy(), (28, 28), interpolation=cv2.INTER_AREA))
    tiles.append(cv2.resize(im[183:296, 389:512].copy(), (28, 28), interpolation=cv2.INTER_AREA))
    tiles.append(cv2.resize(im[303:427, 144:261].copy(), (28, 28), interpolation=cv2.INTER_AREA))
    tiles.append(cv2.resize(im[303:427, 269:382].copy(), (28, 28), interpolation=cv2.INTER_AREA))
    tiles.append(cv2.resize(im[303:427, 389:512].copy(), (28, 28), interpolation=cv2.INTER_AREA))
    for j in range(0, 9):
        if row['POS_'+str(j+1)] == 0:
            train_images = np.concatenate((train_images, tiles[j][np.newaxis, :, :]))
            train_labels = np.append(train_labels, 1)
        elif row['POS_'+str(j+1)] == 1:
            train_images = np.concatenate((train_images, tiles[j][np.newaxis, :, :]))
            train_labels = np.append(train_labels, 0)
        # else:
        #     train_images = np.concatenate((train_images, tiles[j][np.newaxis, :, :]))
        #     train_labels = np.append(train_labels, -1)


# Flip images
train_images_copy = train_images.copy()
train_labels_copy = train_labels.copy()

for i, img in enumerate(train_images_copy):
    if train_labels[i] == 1:
        flipped_img = cv2.flip(img, 0)
        train_images = np.concatenate((train_images, flipped_img[np.newaxis, :, :]))
        train_labels = np.append(train_labels, train_labels_copy[i])
        flipped_img = cv2.flip(img, 1)
        train_images = np.concatenate((train_images, flipped_img[np.newaxis, :, :]))
        train_labels = np.append(train_labels, train_labels_copy[i])
    elif train_labels[i] == 0:
        flipped_img = cv2.flip(img, 0)
        train_images = np.concatenate((train_images, flipped_img[np.newaxis, :, :]))
        train_labels = np.append(train_labels, train_labels_copy[i])
        flipped_img = cv2.flip(img, 1)
        train_images = np.concatenate((train_images, flipped_img[np.newaxis, :, :]))
        train_labels = np.append(train_labels, train_labels_copy[i])


print(train_images.shape,train_labels.shape)

train_images = train_images / 255.0

train_labels = to_categorical(train_labels, 2)

# 1 for x, 0 for o, -1 for blank
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)

#export model
model.save('xo_detection.h5')
# print(train_labels)

# argmax 0=>x
# argmax 1=>o
# argmax 2=>blank