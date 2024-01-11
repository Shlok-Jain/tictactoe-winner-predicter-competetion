import cv2
import numpy as np
from keras.models import load_model
model = load_model('xo_detection.h5')

im = cv2.imread("icg-freshers-data-science-competition/Dataset/Test/0.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# cv2.line(im, (0, 297), (im.shape[1], 297), (127,255,0), 1)
#upper border 58
#lower border 428
#left border 143
#right border 513

#left 1 262
#left 2 268
#left 3 383
#left 4 388

#top 1 177
#top 2 182
#top 3 297
#top 4 302

tiles = []

tiles.append(im[58:177, 143:262].copy())
tiles.append(im[58:177, 268:383].copy())
tiles.append(im[58:177, 388:513].copy())
tiles.append(im[182:297, 143:262].copy())
tiles.append(im[182:297, 268:383].copy())
tiles.append(im[182:297, 388:513].copy())
tiles.append(im[302:428, 143:262].copy())
tiles.append(im[302:428, 268:383].copy())
tiles.append(im[302:428, 388:513].copy())

pred=[]
for i in tiles:
    i = cv2.resize(i, (28, 28), interpolation=cv2.INTER_AREA)  # Resize and preserve aspect ratio
    cv2.imshow("image",i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # i = i / 255.0
    # i = i.reshape((1, 28, 28))
    # predictions = model.predict(i)
    # predicted_label = int(np.argmax(predictions))
    # pred.append(predicted_label)


print(pred)