import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('apple_disease_classifier.h5')
class_names = ['Blotch', 'Normal', 'Rot', 'Scab']

# Load the image
img_path = r"C:\Users\GDS_Manager\Documents\yolov5-master\test_images\post_5d5fb10e60a47.jpg"
labels = r"C:\Users\GDS_Manager\Documents\yolov5-master\runs\detect\exp3\labels\post_5d5fb10e60a47.txt"
img = cv2.imread(img_path)
height, width = img.shape[:2]  # get the height and width of the image

# Load bounding boxes from the file
with open(labels, 'r') as f:
    lines = f.readlines()

bounding_boxes = []
yolo_classes = []
for line in lines:
    values = line.split()
    class_id = int(values[0])
    center_x = float(values[1]) * width
    center_y = float(values[2]) * height
    box_width = float(values[3]) * width
    box_height = float(values[4]) * height
    x_min = int(center_x - box_width / 2)
    y_min = int(center_y - box_height / 2)
    x_max = int(center_x + box_width / 2)
    y_max = int(center_y + box_height / 2)
    bounding_boxes.append((x_min, y_min, x_max, y_max))
    yolo_classes.append(class_id)

# Continue with the rest of the script as before...

for box, yolo_class in zip(bounding_boxes, yolo_classes):
    x_min, y_min, x_max, y_max = box
    apple = img[y_min:y_max, x_min:x_max]
    
    # Resize the apple image to the input size of your classifier model
    apple = cv2.resize(apple, (224, 224))

    # Add an extra dimension to the image tensor for the batch size
    apple_batch = np.expand_dims(apple, axis=0)

    if yolo_class != 0:
        # Predict the class of the apple
        predictions = model.predict(apple_batch)

        # Find the class label with the highest probability
        predicted_class = np.argmax(predictions)

        # Print the predicted class
        print(f'The predicted class for apple is: {class_names[predicted_class]}')
    else:
        predicted_class = 1
    # Draw the bounding box and the predicted class on the image
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(img, class_names[predicted_class], (x_min, y_max+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with bounding boxes and class names
cv2.imshow('Apples', img)
cv2.waitKey(0)
cv2.destroyAllWindows()