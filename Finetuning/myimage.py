
# coding: utf-8
from keras.preprocessing.image import load_img, img_to_array, list_pictures
import numpy as np

def random_crop(image, crop_size):
    height, width = image.shape[:2]
    dy, dx = crop_size
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return image[y:(y+dy), x:(x+dx), :]

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
 
def image_generator(list_of_files, crop_size, to_grayscale=False, scale=1, shift=0):
    while True:
        filename = np.random.choice(list_of_files)
        try:
            img = img_to_array(load_img(filename, to_grayscale))
        except:
            return
        cropped_img = random_crop(img, crop_size)
        
        if cropped_img is None:
            continue
        yield scale * cropped_img - shift
        
def group_by_batch(dataset, batch_size):
    while True:
        try:
            sources, targets = zip(*[next(dataset) for i in xrange(batch_size)])
            batch = (np.stack(sources), np.stack(targets))
            yield batch
        except:
            return
def load_dataset(directory, crop_size, batch_size):
    files = list_pictures(directory)
    generator = image_generator(files, crop_size, scale=1/255.0, shift=0.5)
    generator = group_by_batch(generator, batch_size)
    return generator

