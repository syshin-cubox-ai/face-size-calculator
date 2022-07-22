import csv
import glob
import os

import cv2
import tqdm

import yolo5face_code.yolo5face


if __name__ == '__main__':
    # 데이터셋에 맞게 설정하세요!
    dataset_name = 'helen'
    img_paths = glob.glob('../../data/helen/*')

    # Load detector
    detector = yolo5face_code.yolo5face.YOLO5Face(
        model_cfg='yolo5face_code/yolov5n.yaml',
        model_weights='pth_files/yolov5n-face.pth',
        img_size=640,
        conf_thres=0.3,
        iou_thres=0.5,
        device='cuda',
    )

    with open(f'{dataset_name}_result.csv', mode='w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['image_number', 'image_width', 'image_height', 'image_area',
                         'face_width', 'face_height', 'face_area', 'image_name'])

        for image_number, img_path in enumerate(tqdm.tqdm(img_paths, 'Process images')):
            img = cv2.imread(img_path)
            assert img is not None

            image_name = os.path.basename(img_path)
            image_height, image_width = img.shape[:2]
            image_area = image_width * image_height

            # Detect face
            pred = detector.detect_one(img)

            # Calculate bbox size
            if pred is not None:
                bbox, conf, landmarks = detector.parse_prediction(pred)
                for x1, y1, x2, y2 in bbox:
                    face_width = x2 - x1
                    face_height = y2 - y1
                    face_area = face_width * face_height
                    writer.writerow([image_number, image_width, image_height, image_area,
                                     face_width, face_height, face_area, image_name])
