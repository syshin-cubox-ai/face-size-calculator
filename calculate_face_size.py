import csv
import glob

import cv2
import tqdm

import yolo5face_code.yolo5face


if __name__ == '__main__':
    # 데이터셋에 맞게 설정하세요!
    img_paths = glob.glob('../../data/helen/*')
    criteria = 200 * 200

    # Load detector
    detector = yolo5face_code.yolo5face.YOLO5Face(
        model_cfg='yolo5face_code/yolov5n.yaml',
        model_weights='pth_files/yolov5n-face.pth',
        img_size=640,
        conf_thres=0.3,
        iou_thres=0.5,
        device='cuda',
    )

    with open('result.csv', mode='w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['face number', 'width', 'height', 'area'])

        face_number = 0
        meet_criteria_face_number = 0
        for img_path in tqdm.tqdm(img_paths, 'Process images'):
            # Load image
            img = cv2.imread(img_path)
            assert img is not None

            # Detect face
            pred = detector.detect_one(img)

            # Calculate bbox size
            if pred is not None:
                bbox, conf, landmarks = detector.parse_prediction(pred)
                for x1, y1, x2, y2 in bbox:
                    face_number += 1
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    writer.writerow([face_number, width, height, area])
                    if area < criteria:
                        meet_criteria_face_number += 1
        print(f'Number of faces that meet the criteria: {meet_criteria_face_number}')
