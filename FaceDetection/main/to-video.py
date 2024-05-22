import cv2

# Load the image
img = cv2.imread('image.jpg')  # Assuming the image is in jpg format

# Get the image shape which returns (height, width, channels)
height, width, layers = img.shape

# Define the codec using VideoWriter_fourcc() and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))

# Write the image to the video file (let's do it 300 times to get a 10 second video if your fps is 30)
for i in range(600):
    out.write(img)
    print(f"Progress: {i+1}/600 frames")  # Print the progress

# Release the VideoWriter
out.release()
