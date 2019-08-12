import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.utils import data_utils



class SAIL_EvalDataGenerator(keras.utils.Sequence):
    def __init__(self, data_generator, batch_size = 1):
        self.generator = data_generator
        self.batch_size = batch_size
        self.data_size = self.generator.size()
        self.indices = np.arange(self.data_size)
        self.batch_count = self.data_size//self.batch_size
        if self.batch_count*self.batch_size < self.data_size:
            self.batch_count += 1
        if hasattr(self.generator, 'val_steps'):
            if self.generator.val_steps < self.batch_count:
                self.batch_count = self.generator.val_steps

        self.current = 0
        # self.generated_examples_scales = []
        # self.generated_examples_indices = []

    def __iter__(self):
        return self

    def __len__(self):
        return self.batch_count

    # def __getitem__(self, item):
    #     if item+1 > self.batch_count:
    #         raise StopIteration
    #     else:
    #         images_batch = []
    #         scales_batch = []
    #         left = item * self.batch_size
    #         right = (item + 1) * self.batch_size
    #         if right >= len(self.indices):
    #             right = len(self.indices)
    #         curr_indices_batch = self.indices[left:right]
    #         for idx in curr_indices_batch:
    #             raw_image = self.generator.load_image(idx)
    #             image = self.generator.preprocess_image(raw_image.copy())
    #             image, scale = self.generator.resize_image(image)
    #             if keras.backend.image_data_format() == 'channels_first':
    #                 image = image.transpose((2, 0, 1))
    #
    #             images_batch.append(np.expand_dims(image, 0))
    #             scales_batch.append(scale)
    #         images_batch = np.concatenate(images_batch, axis=0)
    #         # scales_batch = np.array(scales_batch)
    #         # return images_batch,scales_batch,curr_indices_batch
    #
    #         # self.generated_examples_scales = self.generated_examples_scales + scales_batch
    #         # self.generated_examples_indices = self.generated_examples_indices + curr_indices_batch.tolist()
    #         return images_batch


    def __next__(self):
        if self.current+1 > self.batch_count:
            raise StopIteration
        else:
            images_batch = []
            scales_batch = []
            left = self.current * self.batch_size
            right = (self.current + 1) * self.batch_size
            if right >= len(self.indices):
                right = len(self.indices)
            curr_indices_batch = self.indices[left:right]
            self.current += 1
            for idx in curr_indices_batch:
                raw_image = self.generator.load_image(idx)
                image = self.generator.preprocess_image(raw_image.copy())
                image, scale = self.generator.resize_image(image)
                if keras.backend.image_data_format() == 'channels_first':
                    image = image.transpose((2, 0, 1))

                images_batch.append(np.expand_dims(image, 0))
                scales_batch.append(scale)
            images_batch = np.concatenate(images_batch, axis=0)
            return images_batch



    def get_scales(self, steps):
        scales = []
        for step in range(steps):
            left = step * self.batch_size
            right = (step + 1) * self.batch_size
            if right >= len(self.indices):
                right = len(self.indices)
            curr_indices_batch = self.indices[left:right]
            scales_batch = []
            for idx in curr_indices_batch:
                raw_image = self.generator.load_image(idx)
                _, scale = self.generator.resize_image(raw_image)
                scales_batch.append(scale)

            scales = scales+scales_batch
        return scales


    def get_examples_indices(self, steps):
        indices = []
        for step in range(steps):
            left = step * self.batch_size
            right = (step + 1) * self.batch_size
            if right >= len(self.indices):
                right = len(self.indices)
            curr_indices_batch = self.indices[left:right]
            indices_batch = [idx for idx in curr_indices_batch]
            indices = indices+indices_batch
        return indices