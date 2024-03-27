import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import scipy.ndimage as ndi
import argparse
import cv2
import tempfile
import os
import threading

import discorpy.losa.loadersaver as io
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post

# Importing provided functions for distortion correction
from calibration_discorpy import apply_grid_wrap, unwarp_image, numpy_array_to_image_file

class WarpUnwarpApp(tk.Tk):
    def __init__(self, image_path):
        super().__init__()
        self.title("Warp and Unwarp Image Adjuster")
        self.attributes('-fullscreen', True)
        self.attributes("-alpha", 0.5) 
        self.bind("<Escape>", self.exit_fullscreen)

        # Create a full-screen dark overlay frame without the alpha option
        self.overlay_frame = tk.Frame(self, bg='black')
        self.overlay_frame.place(relx=0.25, rely=0.25, relwidth=0.5, relheight=0.5)

        # Loading message label inside the overlay frame
        self.loading_label = ttk.Label(self.overlay_frame, text="Loading...", foreground="white", background="black", font=("Arial", 16))
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")

        # Initially, the overlay and the loading label are not visible
        self.overlay_frame.lower()

        try:
            self.original_image = Image.open(image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            self.destroy()
            return

        self.processed_image = self.original_image.copy()
        self.image_display = ImageTk.PhotoImage(self.original_image)

        # Upper frame for the first two images
        self.top_panel_frame = ttk.Frame(self)
        self.top_panel_frame.pack(side="top", fill="both", expand="yes")

        # Intermediate frame to center the first two images horizontally
        self.intermediate_frame_top = ttk.Frame(self.top_panel_frame)
        self.intermediate_frame_top.pack(expand=True)

        # Label and first image (original)
        self.label_original = ttk.Label(self.intermediate_frame_top, text="Original")
        self.label_original.pack(side="left", padx=(0, 30))  # Space after the label
        self.panel_original = ttk.Label(self.intermediate_frame_top, image=self.image_display)
        self.panel_original.image = self.image_display
        self.panel_original.pack(side="left", expand=True, padx=(0, 50))  # Space after the image

        # Label and second image (warped)
        self.panel_warped = ttk.Label(self.intermediate_frame_top, image=self.image_display)
        self.panel_warped.image = self.image_display
        self.panel_warped.pack(side="left", expand=True)
        self.label_warped = ttk.Label(self.intermediate_frame_top, text="Warped")
        self.label_warped.pack(side="left", padx=(30, 0))  # Space before the label

        # Lower frame for the third image
        self.bottom_panel_frame = ttk.Frame(self)
        self.bottom_panel_frame.pack(side="top", fill="both", expand="yes")

        # Intermediate frame to center the third image vertically
        self.intermediate_frame_bottom = ttk.Frame(self.bottom_panel_frame)
        self.intermediate_frame_bottom.pack(expand=True)

        # Label and third image (unwarped)
        self.label_unwarped = ttk.Label(self.intermediate_frame_bottom, text="Unwarped")
        self.label_unwarped.pack()  # `expand=True` not needed for the label
        self.panel_unwarped = ttk.Label(self.intermediate_frame_bottom, image=self.image_display)
        self.panel_unwarped.image = self.image_display
        self.panel_unwarped.pack(expand=True)

        # Default parameters
        self.setup_default_parameters()

        # Control panel for adjusting parameters
        self.setup_control_panel()

    def setup_control_panel(self):
        self.controls_frame = ttk.Frame(self)
        # Centering the frame horizontally with 'anchor="center"'
        self.controls_frame.pack(side="top", fill="x", expand="yes", anchor="center")

        control_panel = ttk.Frame(self.controls_frame)
        control_panel.pack()

        # Adding sliders and update button within the same frame to align them horizontally
        self.create_adjustment_controls(control_panel)

    def create_adjustment_controls(self, control_panel):
        # Sliders and update button
        self.xcenter_slider = self.create_slider(control_panel, "X Center", 0, self.original_image.width, self.xcenter)
        self.ycenter_slider = self.create_slider(control_panel, "Y Center", 0, self.original_image.height, self.ycenter)
        self.angle_slider = self.create_slider(control_panel, "Angle", -180, 180, self.angle)

        self.coef_sliders = [self.create_slider(control_panel, f"Coef {i}", 0, 1000, coef) for i, coef in enumerate(self.list_coef)]

        self.update_button = ttk.Button(control_panel, text="Update Image", command=self.update_images)
        self.update_button.pack(side="left")

    def display_image(self, img_array, panel, img_type="Original"):
        image_pil = Image.fromarray(img_array.astype(np.uint8))
        # Adjust the maximum size in the 'thumbnail' method to make the images larger
        image_pil.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image_pil)
        panel.config(image=photo)
        panel.image = photo
        panel.config(text=img_type)

    def setup_default_parameters(self):
        self.xcenter = self.original_image.width // 2
        self.ycenter = self.original_image.height // 2
        self.list_coef = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        self.list_power = np.array([1.0, 10**(-4), 10**(-7), 10**(-10), 10**(-13)])
        self.angle = 0

    def create_slider(self, parent, label, from_, to, initial_value):
        slider = tk.Scale(parent, from_=from_, to=to, orient="horizontal", label=label)
        slider.set(initial_value)
        slider.pack(side="left")
        return slider

    def perform_update_task(self):
        # Update the values based on the sliders
        self.xcenter = self.xcenter_slider.get()
        self.ycenter = self.ycenter_slider.get()
        self.angle = self.angle_slider.get()
        self.list_coef = [slider.get() for slider in self.coef_sliders]

        img_np = np.array(self.original_image)

        # Convert the Numpy array to a PIL Image object
        pil_image = Image.fromarray(img_np)

        # Create a temporary file to save the image
        temp_dir = tempfile.gettempdir()  # Get the system's temporary directory
        temp_image_path = os.path.join(temp_dir, "temp_image.png")  # Create a temporary file path

        # Save the image to the temporary file
        pil_image.save(temp_image_path, format="PNG")

        # Apply grid and distortion
        self.apply_effects(temp_image_path)

        # Step 4: Hide the overlay and loading message once done
        self.after(0, self.hide_loading)  # Ensure GUI updates are done on the main thread



    def update_images(self):
       # Show the dark overlay and loading message
        self.overlay_frame.lift()

        # Step 3: Perform the image update task (consider using a separate thread for long tasks)
        threading.Thread(target=self.perform_update_task).start()

        

    def apply_effects(self, image_path):
        mat0 = io.load_image(image_path, average=True)
        mat0 = mat0 / np.max(mat0)
        (height, width) = mat0.shape

        # Create a line-pattern image
        line_pattern = np.zeros((height, width), dtype=np.float32)

        # Horizontal lines
        for i in range(50, height - 50, 40):
            line_pattern[i - 2:i + 3, :] = 1.0

        # Vertical lines
        for j in range(50, width - 50, 40):
            line_pattern[:, j - 2:j + 3] = 1.0

        pad = width // 2  # Need padding as lines are shrunk after warping.
        mat_pad = np.pad(line_pattern, pad, mode='edge')
        mat_pad = ndi.rotate(mat_pad, self.angle, reshape=False)

        warped_image = apply_grid_wrap(self.xcenter, self.ycenter, self.list_coef, self.list_power, mat0, mat_pad, pad)

        img_corrected = unwarp_image(self.xcenter, self.ycenter, self.list_coef, self.list_power, image_path, width, height)
        
        if warped_image.dtype != np.uint8:
            nmin, nmax = np.min(warped_image), np.max(warped_image)
            if nmax != nmin:
                warped_image = np.uint8(255.0 * (warped_image - nmin) / (nmax - nmin))
            else:
                warped_image = np.uint8(warped_image)

        self.display_image(warped_image, self.panel_warped, "Warped")
        self.display_image(img_corrected, self.panel_unwarped, "Unwarped")

    def exit_fullscreen(self, event=None):
        self.attributes('-fullscreen', False)


    def hide_loading(self):
        # Simply lower the overlay frame without specifying what it should be lowered below
        self.overlay_frame.lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warp and Unwarp Image Adjuster")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    app = WarpUnwarpApp(args.image_path)
    app.mainloop()