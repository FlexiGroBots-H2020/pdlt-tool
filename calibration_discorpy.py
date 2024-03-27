import os
import numpy as np
import discorpy.losa.loadersaver as io
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post
import scipy.ndimage as ndi
import cv2

import argparse


def find_point_to_point(points, xcenter, ycenter, list_fact):
    xi, yi = points[1] - xcenter, points[0] - ycenter
    ri = np.sqrt(xi**2 + yi**2)
    factor = np.sum(list_fact * np.power(ri, np.arange(len(list_fact))))
    xo = xcenter + factor * xi
    yo = ycenter + factor * yi
    return xo, yo

def calculate_padding(height, width, xcenter, ycenter, list_ffact):
    corners = [(0, 0), (0, width), (height, 0), (height, width)]
    mapped_corners = [find_point_to_point(corner, xcenter, ycenter, list_ffact) for corner in corners]
    x_mapped, y_mapped = zip(*mapped_corners)

    pad_left = max(-min(x_mapped), 0)
    pad_right = max(max(x_mapped) - width, 0)
    pad_top = max(-min(y_mapped), 0)
    pad_bottom = max(max(y_mapped) - height, 0)

    return int(pad_top), int(pad_bottom), int(pad_left), int(pad_right)

def apply_grid_wrap(xcenter, ycenter, list_coef, list_power, mat0, mat_pad, pad):
    (height, width) = mat0.shape
    list_ffact = list_power * list_coef
    line_img_warped = post.unwarp_image_backward(mat_pad, xcenter + pad,
                                                ycenter + pad, list_ffact)
    line_img_warped = line_img_warped[pad:pad + height, pad:pad + width]

    return (mat0 + 0.5 * line_img_warped)

# Define scanning routines
def scan_coef(xcenter, ycenter, idx, start, stop, step, list_coef, list_power, output_base0, mat0,
              mat_pad, pad, ntime=1, backward=True):
    output_base = output_base0 + "/coef_" + str(idx) + "_ntime_" + str(ntime)
    while os.path.isdir(output_base):
        ntime = ntime + 1
        output_base = output_base0 + "/coef_" + str(idx) + "_ntime_" + str(ntime)
    (height, width) = mat0.shape
    for num in np.arange(start, stop + step, step):
        list_coef_est = np.copy(list_coef)
        list_coef_est[idx] = list_coef_est[idx] + num
        list_ffact = list_power * list_coef_est
        line_img_warped = post.unwarp_image_backward(mat_pad, xcenter + pad,
                                                     ycenter + pad, list_ffact)
        line_img_warped = line_img_warped[pad:pad + height, pad:pad + width]
        name = "coef_{0}_val_{1:4.2f}".format(idx, list_coef_est[idx])
        io.save_image(output_base + "/forward/img_" + name + ".jpg", mat0 + 0.5 * line_img_warped)
        if backward is True:
            # Transform to the backward model for correction
            hlines = np.int16(np.linspace(0, height, 40))
            vlines = np.int16(np.linspace(0, width, 50))
            ref_points = [[i - ycenter, j - xcenter] for i in hlines for j in vlines]
            list_bfact = proc.transform_coef_backward_and_forward(list_ffact, ref_points=ref_points)
            img_unwarped = post.unwarp_image_backward(mat0, xcenter, ycenter, list_bfact)
            io.save_image(output_base + "/backward/img_" + name + ".jpg", img_unwarped)

def scan_center(xcenter, ycenter, start, stop, step, list_coef, list_power,
                output_base0, mat0, mat_pad, pad, axis="x", ntime=1, backward=True):
    output_base = output_base0 + "/" + axis + "_center" + "_ntime_" + str(ntime)
    while os.path.isdir(output_base):
        ntime = ntime + 1
        output_base = output_base0 + "/" + axis + "_center" + "_ntime_" + str(ntime)
    (height, width) = mat0.shape
    list_ffact = list_power * list_coef
    if axis == "x":
        for num in np.arange(start, stop + step, step):
            line_img_warped = post.unwarp_image_backward(mat_pad,
                                                         xcenter + num + pad,
                                                         ycenter + pad,
                                                         list_ffact)
            line_img_warped = line_img_warped[pad:pad + height, pad:pad + width]
            name = "xcenter_{0:7.2f}".format(xcenter + num)
            io.save_image(output_base + "/forward/img_" + name + ".jpg", mat0 + 0.5 * line_img_warped)
            if backward is True:
                # Transform to the backward model for correction
                hlines = np.int16(np.linspace(0, height, 40))
                vlines = np.int16(np.linspace(0, width, 50))
                ref_points = [[i - ycenter, j - xcenter] for i in hlines for j in vlines]
                list_bfact = proc.transform_coef_backward_and_forward(list_ffact, ref_points=ref_points)
                img_unwarped = post.unwarp_image_backward(mat0, xcenter+num, ycenter, list_bfact)
                io.save_image(output_base + "/backward/img_" + name + ".jpg",  img_unwarped)
    else:
        for num in np.arange(start, stop + step, step):
            line_img_warped = post.unwarp_image_backward(mat_pad, xcenter + pad,
                                                         ycenter + num + pad,
                                                         list_ffact)
            line_img_warped = line_img_warped[pad:pad + height, pad:pad + width]
            name = "ycenter_{0:7.2f}".format(ycenter + num)
            io.save_image(output_base + "/forward/img_" + name + ".jpg", mat0 + 0.5 * line_img_warped)
            if backward is True:
                # Transform to the backward model for correction
                hlines = np.int16(np.linspace(0, height, 40))
                vlines = np.int16(np.linspace(0, width, 50))
                ref_points = [[i - ycenter, j - xcenter] for i in hlines for j in vlines]
                list_bfact = proc.transform_coef_backward_and_forward(list_ffact, ref_points=ref_points)
                img_unwarped = post.unwarp_image_backward(mat0, xcenter, ycenter+num, list_bfact)
                io.save_image(output_base + "/backward/img_" + name + ".jpg",  img_unwarped)

def unwarp_image(xcenter, ycenter, list_coef, list_power, image_path, width, height):
    # Get a good estimation of the forward model
    list_ffact = list_coef * list_power

    # Transform to the backward model for correction
    ref_points = [[i - ycenter, j - xcenter] for i in range(0, height, 50) for j in
                range(0, width, 50)]
    list_bfact = proc.transform_coef_backward_and_forward(list_ffact, ref_points=ref_points)

    # Find top-left point in the undistorted space given top-left point in the distorted space.
    xu_top_left, yu_top_left = find_point_to_point((0, 0), xcenter, ycenter, list_ffact)
    # Find bottom-right point in the undistorted space given bottom-right point in the distorted space.
    xu_bot_right, yu_bot_right = find_point_to_point((height - 1, width - 1), xcenter, ycenter,
                                                    list_ffact)

    # Calculate padding width for each side.
    pad_top = max(int(np.abs(yu_top_left)), 0)
    pad_bot = max(int(yu_bot_right - height), 0)
    pad_left = max(int(np.abs(xu_top_left)), 0)
    pad_right = max(int(xu_bot_right - width), 0)


    img = io.load_image(image_path, average=False)
    img_pad = np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)), mode="constant")
    img_corrected = unwarp_color_image(img_pad, xcenter + pad_left, ycenter + pad_top,
                                    list_bfact, mode='constant')
    return img_corrected

 # Define a function for convenient use.
def unwarp_color_image(img, xcenter, ycenter, list_bfact, mode='reflect'):
    if len(img.shape) != 3:
        raise ValueError("Check if input is a color image!!!")
    img_corrected = []
    for i in range(img.shape[-1]):
        img_corrected.append(post.unwarp_image_backward(img[:, :, i], xcenter,
                                                        ycenter, list_bfact, mode=mode))
    img_corrected = np.moveaxis(np.asarray(img_corrected), 0, 2)
    return img_corrected


def main(image_path, output_base = "output/distortion/"):

    # Check if the output directory exists, and if not, create it
    if not os.path.exists(output_base):
        os.makedirs(output_base)  

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


    # Estimate parameters by visual inspection:
    # Coarse estimation
    xcenter = width // 2 
    ycenter = height // 2 
    list_power = np.asarray([1.0, 10**(-4), 10**(-7), 10**(-10), 10**(-13)])
    list_coef = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0])

    # Rotate the line-pattern image if need to
    angle = -5.0 # Degree
    pad = width // 2 # Need padding as lines are shrunk after warping.
    mat_pad = np.pad(line_pattern, pad, mode='edge')
    mat_pad = ndi.rotate(mat_pad, angle, reshape=False)

    ## Scan the 4th coefficient
    #scan_coef(xcenter, ycenter, 4, 0, 50, 1, list_coef, list_power, output_base, mat0, mat_pad, pad)
    ## The value of 24.0 is good, update the 4th coefficient.
    list_coef[4] = 94.0

    ## Scan the 3rd coefficient
    #scan_coef(xcenter, ycenter, 3, 0, 50, 1, list_coef, list_power, output_base, mat0, mat_pad, pad)
    ## The value of 2.0 is good, update the 3rd coefficient.
    list_coef[3] = 68.0

    ## Scan the 2nd coefficient
    #scan_coef(xcenter, ycenter, 2, 0, 50, 1, list_coef, list_power, output_base, mat0, mat_pad, pad)
    ## The value of 5.0 is good, update the 2nd coefficient.
    list_coef[2] = 25.0

    ## Scan the x-center
    #scan_center(xcenter, ycenter, -150, 150, 2, list_coef, list_power, output_base,
    #            mat0, mat_pad, pad, axis="x")
    ## Found x=648 looks good.
    #xcenter = 390

    ## Scan the y-center
    #scan_center(xcenter, ycenter, -150, 150, 2, list_coef, list_power, output_base,
    #            mat0, mat_pad, pad, axis="y")
    ## Found y=480 looks good.
    #ycenter = 160

    # Adjust the 1st-order and 0-order coefficients manually if need to.
    list_coef[1] = 1.0
    list_coef[0] = 1.0

    img_corrected = unwarp_image(xcenter, ycenter, list_coef, list_power, image_path, width, height)
    io.save_image(output_base + "/unwarped_padding.jpg", img_corrected[:,:,:3])

    img_aux = apply_grid_wrap(xcenter, ycenter, list_coef, list_power, mat0, mat_pad, pad)
    io.save_image(output_base + "/aux_padding.jpg",  img_aux)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image for distortion correction.")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--output_base", type=str, default="output/distortion/", help="Base path for output")
    
    args = parser.parse_args()

    main(args.image_path, args.output_base)
