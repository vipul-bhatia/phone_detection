import cv2
import os

# Set the path to the folder containing images
image_folder_path = 'saved_frames'

# Get the list of image files in the folder
image_files = [file for file in os.listdir(image_folder_path) if file.endswith('.jpg')]

# Sort the image files based on their numerical order in the filename
image_files.sort(key=lambda x: int(x.split('.')[0]))

# Set the video output file path
output_video_path = 'output_video.avi'

# Get the first image to extract its dimensions
first_image_path = os.path.join(image_folder_path, image_files[0])
first_image = cv2.imread(first_image_path)
height, width, layers = first_image.shape

# Create a VideoWriter object
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 1, (width, height))

# Loop through each image and add it to the video
for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    frame = cv2.imread(image_path)

    # Write the frame to the video
    video_writer.write(frame)

    # Display the frame with a wait key of 2 second
    cv2.imshow('Video', frame)
    cv2.waitKey(4000)  # 1000 milliseconds (2 second)

# Release the VideoWriter object
video_writer.release()

# Destroy the OpenCV window
cv2.destroyAllWindows()
