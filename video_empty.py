import numpy as np
import cv2
import os 

class Drainer:
    def __init__(self, height, width):
        # Initialize the empty_image with random noise.
        self.empty_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        self.prev_frame = None  # Initialize prev_frame attribute to store the previous frame
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def detect_shadows(self, frame):
          
        # Apply the background subtractor to get the foreground mask
        fgmask = self.fgbg.apply(frame)
        
        # Shadows are marked in gray (value 127). Convert to binary mask (255) for shadows, 0 for everything else
        _, thresh = cv2.threshold(fgmask, 127,255,cv2.THRESH_BINARY)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        kernel = np.ones((10, 10), np.uint8)

        thresh = cv2.dilate(thresh,  np.ones((15, 15), np.uint8), iterations=3)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
        return cv2.bitwise_not(thresh)

    def update_empty_image(self, frame, masks, base_alpha=0.2, variance_threshold=1000):
        new_frame = np.array(frame, dtype=np.uint8)
        frame_variance = np.var(new_frame)

        if self.prev_frame is None:
            self.prev_frame = new_frame  # If it's the first frame, just update prev_frame
            return  # No further processing needed for the first frame

        # Convert frames to grayscale for motion detection
        gray_new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        gray_prev_frame = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(gray_prev_frame, gray_new_frame)

        # Apply a threshold to highlight areas with significant differences
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

        # Dilate the thresholded image to fill in gaps
        dilated_thresh = cv2.dilate(thresh, np.ones((5,5), np.uint8), iterations=3)

        # Use dilated_thresh as a mask to exclude areas with motion from the update
        #motion_mask = cv2.cvtColor(dilated_thresh, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels to match empty_image
        motion_mask = dilated_thresh

        # Adjust alpha based on frame variance
        if frame_variance < variance_threshold:
            # Low variance, the frame is similar to previous ones, so give it more weight
            alpha = min(base_alpha * 1.5, 1.0)  # Ensuring alpha doesn't exceed 1
        else:
            # High variance, the frame differs significantly, so reduce its impact
            alpha = max(base_alpha * 0.5, 0.1)  # Ensuring alpha doesn't drop below 0.1

        # Initialize a base mask with the same dimensions as the frame, but with all values set to 255 (white).
        mask_base = np.zeros(new_frame.shape[:2], dtype="uint8") * 255

        for ii, mask in enumerate(masks):
            mask_np = mask.cpu().numpy().astype("uint8")*255  # Convert the mask to a NumPy array.
            # Define the kernel size for dilation. The size and shape of the kernel will affect the amount of dilation.
            # A larger kernel will result in a more significant dilation.
            kernel = np.ones((5, 5), np.uint8)

            # Apply dilation to the mask. The 'iterations' parameter controls how many times the operation is applied.
            dilated_mask = cv2.dilate(mask_np, kernel, iterations=2)

            # This adds the current object's mask to the overall mask, excluding it from being updated in the empty_image.
            mask_base = cv2.bitwise_or(mask_base, dilated_mask)

        # Invert the combined mask to create the update mask (objects to be excluded are now black).
        update_mask = cv2.bitwise_not(mask_base)

        # invert the mask
        motion_mask = cv2.bitwise_not(motion_mask)

        # include shadows check 
        shadow_mask = self.detect_shadows(frame)

        # Merge motion_mask and update_mask
        #merge_mask = cv2.bitwise_and(motion_mask, update_mask, shadow_mask)
        merge_mask = cv2.bitwise_and(motion_mask, update_mask)
        merge_mask = cv2.bitwise_and(merge_mask, shadow_mask)

        kernel = np.ones((10, 10), np.uint8)

        # Apply dilation to the mask. The 'iterations' parameter controls how many times the operation is applied.
        dilated_mask = cv2.erode(merge_mask, kernel, iterations=2) # inverse logic because the mask is OR

        cv2.imshow("shadows", dilated_mask)

        # Create a 3-channel version of the update mask to work with the color image.
        update_mask_3ch = cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR)

        # Define the weights for the existing content and the new frame.
        # Alpha is the weight for the new frame, and (1 - Alpha) will be the weight for the existing content in empty_image.
        # Setting Alpha < 0.5 will ensure that the new frame has less influence than the existing content.

        # Update the empty_image using a weighted average of the current empty_image and new_frame,
        # but only in regions not covered by the update mask.
        self.empty_image = np.where(update_mask_3ch == 0, self.empty_image, ((1 - alpha) * self.empty_image + alpha * new_frame).astype(np.uint8))

        # Update prev_frame for the next iteration
        self.prev_frame = new_frame

    def get_empty_image(self):
        # Return the current state of the empty_image.
        return self.empty_image

