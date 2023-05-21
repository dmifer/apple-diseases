import os
import cv2

def save_detected_objects(txt_folder, source_folder, output_folder):
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for txt_file in txt_files:
        image_name = txt_file.replace('.txt', '.jpg')
        image_path = os.path.join(source_folder, image_name)
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        with open(os.path.join(txt_folder, txt_file)) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                data = line.strip().split(' ')
                class_id, x_center, y_center, width, height = map(float, data)
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                cropped = image[y1:y2, x1:x2]
                output_path = os.path.join(output_folder, f"{image_name.split('.')[0]}_object_{i + 1}.jpg")
                cv2.imwrite(output_path, cropped)

txt_folder = r"C:\Users\GDS_Manager\Documents\yolov5-master\runs\detect\exp2\labels"
source_folder = r"C:\Users\GDS_Manager\Documents\yolov5-master\test_images"
output_folder = r"C:\Users\GDS_Manager\Documents\yolov5-master\cut_result"

save_detected_objects(txt_folder, source_folder, output_folder)