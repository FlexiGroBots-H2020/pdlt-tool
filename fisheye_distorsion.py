from defisheye import Defisheye

dtype = 'linear'
format = 'fullframe'
fov = 100
pfov = 90

img = "runs/track/exp30/empty_frame.png"
img_out = f"./example_{dtype}_{format}_{pfov}_{fov}.jpg"

obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)

# To save image locally 
obj.convert(outfile=img_out)

# To use the converted image in memory

new_image = obj.convert()