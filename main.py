# import cv2
# from YOLOv7 import YOLOv7
# from utils import draw_detections

# # Initialize video
# video_path = "1.mp4"
# cap = cv2.VideoCapture(video_path)
# start_time = 0  # skip first {start_time} seconds
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)

# # Initialize YOLOv7 model with GPU support
# model_path = "yolov7_384x640.onnx"
# yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)

# cv2.namedWindow("Detected Cell Phones", cv2.WINDOW_NORMAL)
# while cap.isOpened():

#     # Press key q to stop
#     if cv2.waitKey(1) == ord('q'):
#         break

#     try:
#         # Read frame from the video
#         ret, frame = cap.read()
#         if not ret:
#             break
#     except Exception as e:
#         print(e)
#         continue

#     # Update object localizer
#     boxes, scores, class_ids = yolov7_detector(frame)

#     # Filter out only cell phones (class ID 67)
#     cell_phone_indices = [i for i, class_id in enumerate(class_ids) if class_id == 67]
#     cell_phone_boxes = [boxes[i] for i in cell_phone_indices]
#     cell_phone_scores = [scores[i] for i in cell_phone_indices]

#     # if cell_phone_scores != None:

#     #     cv2.putText(frame, "Driver using Phone", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

#     # Draw bounding boxes only for cell phones
#     detection_img = draw_detections(frame, cell_phone_boxes, cell_phone_scores, class_ids=class_ids)

#     # Display the result
#     cv2.imshow("Detected Cell Phones", detection_img)

#     # Check if cell phones are detected and print a message
#     if cell_phone_indices:
#         print("Phone detected in the frame")

# # Release resources
# cap.release()
# cv2.destroyAllWindows()







# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from io import BytesIO
# import cv2
# import numpy as np
# import os
# from datetime import datetime

# app = FastAPI()

# origins = ["http://localhost:3000"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Create a folder to save the frames
# save_folder = "saved_frames"
# os.makedirs(save_folder, exist_ok=True)

# # Counter for serial naming
# frame_counter = 1

# # Endpoint to process the received frame using an OpenCV model
# @app.post("/api/process_frame/")
# async def process_frame(file: UploadFile = File(...)):
#     global frame_counter

#     # Read the uploaded frame
#     contents = await file.read()

#     # Decode the frame
#     nparr = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # Perform your OpenCV model processing on the frame
#     # ...

#     # Example: Convert the frame to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Save the processed frame to a file with a serial name
#     filename = f"{save_folder}/{frame_counter}.jpg"
#     cv2.imwrite(filename, gray_frame)

#     # Increment the frame counter for the next frame
#     frame_counter += 1

#     # Return the processed frame
#     _, buffer = cv2.imencode('.jpg', gray_frame)
#     processed_frame_bytes = buffer.tobytes()

#     return StreamingResponse(BytesIO(processed_frame_bytes), media_type="image/jpeg")



from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import cv2
import numpy as np
import os
from datetime import datetime
from YOLOv7 import YOLOv7
from utils import draw_detections

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a folder to save the frames
save_folder = "saved_frames"
os.makedirs(save_folder, exist_ok=True)

# Counter for serial naming
frame_counter = 1

# Initialize YOLOv7 model with GPU support
model_path = "yolov7_384x640.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)

# Function to perform phone detection on a given frame
def detect_phones(frame):
    # Perform phone detection using YOLOv7 model
    boxes, scores, class_ids = yolov7_detector(frame)

    # Filter out only cell phones (class ID 67)
    cell_phone_indices = [i for i, class_id in enumerate(class_ids) if class_id == 67]
    cell_phone_boxes = [boxes[i] for i in cell_phone_indices]
    cell_phone_scores = [scores[i] for i in cell_phone_indices]

    # Draw bounding boxes only for cell phones
    detection_img = draw_detections(frame, cell_phone_boxes, cell_phone_scores, class_ids=class_ids)

    # Display the result
    cv2.imshow("Detected Cell Phones", detection_img)

    # Check if cell phones are detected and print a message
    if cell_phone_indices:
        print("Phone detected in the frame")

    return detection_img

# Endpoint to process the received frame using YOLOv7 for phone detection
@app.post("/api/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    global frame_counter

    # Read the uploaded frame
    contents = await file.read()

    # Decode the frame
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform phone detection
    detected_frame = detect_phones(frame)

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2GRAY)

    # Save the processed frame to a file with a serial name
    filename = f"{save_folder}/{frame_counter}.jpg"
    cv2.imwrite(filename, gray_frame)

    # Increment the frame counter for the next frame
    frame_counter += 1

    # Return the processed frame
    _, buffer = cv2.imencode('.jpg', gray_frame)
    processed_frame_bytes = buffer.tobytes()

    return StreamingResponse(BytesIO(processed_frame_bytes), media_type="image/jpeg")
