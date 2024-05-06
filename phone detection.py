import cv2
from YOLOv7 import YOLOv7
from utils import draw_detections

# Initialize video
video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)
start_time = 0  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)

# Initialize YOLOv7 model with GPU support
model_path = "yolov7_384x640.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Cell Phones", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov7_detector(frame)

    # Filter out only cell phones (class ID 67)
    cell_phone_indices = [i for i, class_id in enumerate(class_ids) if class_id == 67]
    cell_phone_boxes = [boxes[i] for i in cell_phone_indices]
    cell_phone_scores = [scores[i] for i in cell_phone_indices]

    # if cell_phone_scores != None:

    #     cv2.putText(frame, "Driver using Phone", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    # Draw bounding boxes only for cell phones
    detection_img = draw_detections(frame, cell_phone_boxes, cell_phone_scores, class_ids=class_ids)

    # Display the result
    cv2.imshow("Detected Cell Phones", detection_img)

    # Check if cell phones are detected and print a message
    if cell_phone_indices:
        print("Phone detected in the frame")

# Release resources
cap.release()
cv2.destroyAllWindows()




