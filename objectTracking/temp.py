from ultralytics import YOLO
import cv2
import math
import random
from tracker import Tracker

cap=cv2.VideoCapture(0)

frame_width=int(cap.get(3))
frame_height = int(cap.get(4))

out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

model=YOLO("yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
while True:
    success, img = cap.read()
    # Doing detections using YOLOv8 frame by frame
    #stream = True will use the generator and it is more efficient than normal
    results=model(img,stream=True)
    #Once we have the results we can check for individual bounding boxes and see how well it performs
    # Once we have have the results we will loop through them and we will have the bouning boxes for each of the result
    # we will loop through each of the bouning box
    for r in results:
        # boxes=r.boxes
        detections = []
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            #print(x1, y1, x2, y2)
            x1,y1,x2,y2, score, class_id = int(x1), int(y1), int(x2), int(y2),int(score), int(class_id)
            detections.append([x1, y1, x2, y2, score])
            print(x1,y1,x2,y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
            tracker.update(img, detections)
            #print(box.conf[0])
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            class_name=classNames[cls]
            label=f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]


            #print(t_size)
            # c2 = x1 + t_size[0], y1 - t_size[1] - 3
            # cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
            # cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)


            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)



    out.write(img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('1'):
        break
out.release()