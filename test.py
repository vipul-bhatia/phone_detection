import cv2
import onnx
import onnxruntime
import numpy as np

# Constants
depth_threshold = 50

# Load MiDaS ONNX model
onnx_model_path = "midas.onnx"
onnx_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# Input dimensions for MiDaS (adjust accordingly)
input_height, input_width = 256, 256

# Specify the path to the video file
video_path = 'vid2.mp4'
cap = cv2.VideoCapture(video_path)

output_width = 800
output_height = 600

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Resize and normalize the input frame
    input_frame = cv2.resize(frame, (input_width, input_height))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = input_frame.astype(np.float32) / 255.0

    # Reshape the input frame for ONNX model
    input_data = input_frame.transpose(2, 0, 1).reshape(1, 3, input_height, input_width)

    # Make a prediction using MiDaS ONNX model
    depth_output = onnx_session.run(None, {'input': input_data})[0].squeeze()

    # Resize the frame and depth map
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Convert depth map to 3-channel image using colormap
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_output, alpha=255.0 / np.max(depth_output)),
        cv2.COLORMAP_JET
    )

    resized_depth = cv2.resize(depth_colormap, (output_width, output_height))

    cv2.imshow('Depth Estimation', np.hstack((resized_frame, resized_depth)))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()