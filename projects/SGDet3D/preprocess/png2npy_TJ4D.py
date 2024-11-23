import os
import numpy as np
import cv2
from PIL import Image
import concurrent.futures
def get_downsample_depths_numpy(depth, down, processing='min'):
    H, W = depth.shape
    depth = depth.reshape(H//down, down, W//down, down, 1)
    depth = depth.transpose(0, 2, 4, 1, 3)
    depth = depth.reshape(-1, down * down)
    depth_tmp = np.where(depth == 0.0, 1e5, depth)
    if processing == 'min': 
        depth = np.min(depth_tmp, axis=-1)
    if processing == 'max': 
        depth = np.max(depth_tmp, axis=-1)
    if processing == 'mean': 
        depth = np.mean(depth_tmp, axis=-1)
    depth = depth.reshape(H//down, W//down)
    # cv2.imwrite('depth.png', depth)
    return depth

def get_upsample_depths_numpy(depth, up):
    H, W = depth.shape
    depth = np.repeat(depth, up, axis=0)
    depth = np.repeat(depth, up, axis=1)
    # cv2.imwrite('depth.png', depth)
    return depth

def method_upsample(image, scale_factor):
    new_size = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    # image = get_upsample_depths_numpy(image, 2)
    return image
def method_padzeors(image, crop_top_pixels):
    residual = np.ones((crop_top_pixels, image.shape[1]))*0.0 # *80.0
    image = np.concatenate([residual, image], axis=0) # three channel the same
    return image
    
def process_image(filename, input_dir, otput_dir, image_type='segmt', method=None, **kargs):
    path = os.path.join(input_dir, filename)
    if image_type=='depth': image = np.array(Image.open(path), dtype=np.float32) / 255.0
    if image_type=='segmt': image = np.array(Image.open(path), dtype=np.bool_)
    image = np.array(Image.open(path), dtype=np.float32) / 255.0
    if method is not None: image = method(image, **kargs)
    save_path = os.path.join(otput_dir, os.path.splitext(filename)[0] + '.npy')
    np.save(save_path, image)
    return filename
            
def processing(input_dir, otput_dir, image_type='segmt', method=None, **kargs):
    if not os.path.exists(otput_dir): os.makedirs(otput_dir)
    list_name = os.listdir(input_dir)
    list_name.sort()
    list_name = [name for name in list_name if name.endswith(".png")]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, filename, input_dir, otput_dir, image_type, method, **kargs) for filename in list_name]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            filename = future.result()
            print('Processed %s %d/%d' % (filename, i + 1, len(list_name)))
            
if __name__ == '__main__':
    # processing segmentation
    input_dir = 'data/TJ4D/segmentation'
    otput_dir = 'data/TJ4D/segmentation'
    if not os.path.exists(otput_dir): os.makedirs(otput_dir)
    processing(input_dir, otput_dir, image_type='segmt', method=None)
    segmt = np.load(os.path.join(otput_dir, '000001.npy'))
    cv2.imwrite('segmentation.png', 255.0*segmt)

