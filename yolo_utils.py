

def scale_masks(im0, masks):
    
    masks_im0 = []
    for m in masks:
        h,w = im0.shape()
        #.cpu().numpy().astype("int8")
        masks_im0.append(m[0])
        
    return masks_im0