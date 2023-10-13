import cv2
import tensorflow as tf

#using pretrained model 
model = tf.saved_model.load('ssd_mobilenet_v1_coco_2017_11_17/saved_model')

# Preprocess the image
img = cv2.imread('image.jpg')
img = cv2.resize(img, (300, 300))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run the model on the image
detections = model.predict(tf.expand_dims(img, axis=0))

for detection in detections[0]:
    bbox = detection['bbox']
    score = detection['scores'][0]

    if score > 0.5:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)