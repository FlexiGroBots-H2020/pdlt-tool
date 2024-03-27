import cv2
import numpy as np
import os

def extract_3_frame_snippets(video_path, frame_numbers, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi format

    # Process each frame number
    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)  # Go to the frame before the target
        success, prev_frame = cap.read()
        if not success:
            continue  # Skip if the frame cannot be read
        
        success, target_frame = cap.read()
        if not success:
            continue  # Skip if the frame cannot be read
        
        success, next_frame = cap.read()
        if not success:
            continue  # Skip if the frame cannot be read
        
        # Create a video writer object
        output_file_path = os.path.join(output_dir, f"snippet_around_frame_{frame_number}.mp4")
        out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

        # Write the frames to the output file
        out.write(prev_frame)
        out.write(target_frame)
        out.write(next_frame)

        # Release the video writer
        out.release()
    
    # Release the video capture
    cap.release()

# Example usage:
video_path = 'input/jere/dept_validation.mp4'
frame_numbers = [90, 160, 310]  # Replace with your frame numbers
output_dir = 'output'

extract_3_frame_snippets(video_path, frame_numbers, output_dir)

