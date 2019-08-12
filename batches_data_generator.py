import numpy as np
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

class SAIL_batches_generator:
    def __init__(self, data_crops, batch_size = 32):
        self.data = data_crops
        self.batch_size = batch_size
        self.batches = self.data.shape[0]//self.batch_size
        if self.batch_size * self.batches < self.data.shape[0]:
            self.batches += 1
        self.current = 0

    def __next__(self):
        if self.current+1 > self.batches:
            raise StopIteration('there is no more batches!')
        curr_data_batch = self.data[self.current*self.batch_size:(self.current+1)*self.batch_size]
        processed_images = []
        scales = []
        for image in curr_data_batch:
            processed_image = image*255.
            processed_image = preprocess_image(processed_image)
            processed_image, scale = resize_image(processed_image, min_side=512)
            processed_images.append(np.expand_dims(processed_image, 0))
            scales.append(scale)
        processed_images = np.concatenate(processed_images, axis=0)

        self.current += 1
        return processed_images, scales

    def __len__(self):
        return self.batches