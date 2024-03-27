import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import scipy.ndimage as ndi
import argparse

# Importing your provided functions here
from calibration_discorpy import apply_grid_wrap, unwarp_image

class WarpUnwarpApp(tk.Tk):
    def __init__(self, image_path):
        super().__init__()
        self.title("Warp and Unwarp Image Adjuster")

        # Establecer la aplicación en pantalla completa
        self.attributes('-fullscreen', True)

        # Salir de pantalla completa con la tecla Esc
        self.bind("<Escape>", self.exit_fullscreen)

        # Load and process the image
        self.original_image = Image.open(image_path)
        self.processed_image = self.original_image.copy()
        self.image_display = ImageTk.PhotoImage(self.original_image)

        # Image Panels
        self.panel_frame = ttk.Frame(self)
        self.panel_frame.pack(side="top", fill="both", expand="yes")

        self.panel_original = ttk.Label(self.panel_frame, image=self.image_display)
        self.panel_original.image = self.image_display
        self.panel_original.pack(side="left", fill="both", expand="yes")

        self.panel_warped = ttk.Label(self.panel_frame, image=self.image_display)
        self.panel_warped.image = self.image_display
        self.panel_warped.pack(side="left", fill="both", expand="yes")

        self.panel_unwarped = ttk.Label(self.panel_frame, image=self.image_display)
        self.panel_unwarped.image = self.image_display
        self.panel_unwarped.pack(side="left", fill="both", expand="yes")

        # Default parameters
        self.xcenter = self.original_image.width // 2
        self.ycenter = self.original_image.height // 2
        self.list_coef =  np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        self.list_power = np.array([1.0, 10**(-4), 10**(-7), 10**(-10), 10**(-13)])
        self.angle = 0

        # Control Panel for parameters
        self.controls_frame = ttk.Frame(self)
        self.controls_frame.pack(side="top", fill="both", expand="yes")

        self.create_adjustment_controls()

    def exit_fullscreen(self, event=None):
        self.attributes('-fullscreen', False)

    def create_adjustment_controls(self):
        # Sliders for parameters
        control_panel = ttk.Frame(self.controls_frame)
        control_panel.pack()

        self.xcenter_slider = tk.Scale(control_panel, from_=0, to=self.original_image.width, orient="horizontal", label="X Center")
        self.xcenter_slider.set(self.xcenter)
        self.xcenter_slider.pack(side="left")

        self.ycenter_slider = tk.Scale(control_panel, from_=0, to=self.original_image.height, orient="horizontal", label="Y Center")
        self.ycenter_slider.set(self.ycenter)
        self.ycenter_slider.pack(side="left")

         # Dynamically create sliders for each coefficient
        self.coef_sliders = []
        for i, coef in enumerate(self.list_coef):
            slider = tk.Scale(self.controls_frame, from_=0, to=1000, resolution=1, orient="horizontal", label=f"Coef {i}")
            slider.set(coef)
            slider.pack()
            self.coef_sliders.append(slider)

        # Slider for angle
        self.angle_slider = tk.Scale(control_panel, from_=-180, to=180, orient="horizontal", label="Angle")
        self.angle_slider.set(self.angle)  # Asegúrate de haber definido self.angle en __init__
        self.angle_slider.pack(side="left")

        # Update Button
        self.update_button = ttk.Button(self.controls_frame, text="Update Image", command=self.update_images)
        self.update_button.pack()

    def update_images(self):
        if hasattr(self, 'original_image'):
            # Update xcenter, ycenter, and coefficients based on sliders' values
            self.xcenter = self.xcenter_slider.get()
            self.ycenter = self.ycenter_slider.get()
            self.angle = self.angle_slider.get()
            self.list_coef = [slider.get() for slider in self.coef_sliders]
            # Convert PIL Image to numpy array for processing
            img_np = np.array(self.original_image.convert('L'))  # Convert to grayscale for processing

            # Create a line-pattern image for the grid
            line_pattern = np.zeros_like(img_np)
            for i in range(0, line_pattern.shape[0], 40):
                line_pattern[i:i+2, :] = 255  # Horizontal lines
            for j in range(0, line_pattern.shape[1], 40):
                line_pattern[:, j:j+2] = 255  # Vertical lines

            # Rotate the line-pattern image if needed
            angle = -5.0  # Degree
            pad = max(line_pattern.shape) // 2  # Padding
            mat_pad = np.pad(line_pattern, pad, mode='edge')
            mat_pad = ndi.rotate(mat_pad, angle, reshape=False)

            # Apply grid and warp
            warped_image = apply_grid_wrap(self.xcenter, self.ycenter, self.list_coef, self.list_power, img_np, mat_pad, pad)
            warped_image_pil = Image.fromarray(warped_image.astype(np.uint8))  # Convert to PIL Image for display
            self.display_image(warped_image_pil, self.panel_warped)

            # Assuming unwarp_image function handles color image unwrapping
            # You might need to adjust this part to work with your actual unwarp_image implementation
            unwarped_image = unwarp_image(self.xcenter, self.ycenter, self.list_coef, self.list_power, img_np, img_np.shape[1], img_np.shape[0])
            unwarped_image_pil = Image.fromarray(unwarped_image.astype(np.uint8))  # Convert to PIL Image for display
            self.display_image(unwarped_image_pil, self.panel_unwarped)


    def display_image(self, image, panel):
        # Usa Image.Resampling.LANCZOS en lugar de Image.ANTIALIAS
        image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        panel.config(image=photo)
        panel.image = photo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warp and Unwarp Image Adjuster")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    app = WarpUnwarpApp(args.image_path)
    app.mainloop()
