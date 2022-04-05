import tensorflow as tf
import numpy as np
import pandas as pd
from . import hic_loader, chip_loader

## Code modified from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly#data-generator
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, regions_df, chip_names,
                cool_path, chip_path_dict,
                cool_resolution, chip_resolution,
                batch_size=32, upper_triangle=True, shuffle=True):
        'Initialization'

        self.regions_df = regions_df
        self.region_len = int(regions_df.iloc[0].end - regions_df.iloc[0].start)
        self.chip_names = chip_names
        self.cool_path = cool_path
        self.chip_path_dict = chip_path_dict
        self.cool_resolution = cool_resolution
        self.chip_resolution = chip_resolution
        self.hic_dim = (self.region_len//cool_resolution, self.region_len//cool_resolution)
        self.upper_indices = np.triu_indices(self.hic_dim[0])
        self.chip_dim = (self.region_len//chip_resolution, len(chip_names))
        self.upper_triangle = upper_triangle

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.regions_df) / self.batch_size))
    
    def get_generator_dict(self, regions_df, chip_names, chip_path_dict):
        generator_dict = {}

        for name in chip_names:
            path = chip_path_dict[name]
            generator_dict[name] = chip_loader(regions_df, path, resolution=self.chip_resolution)

        return generator_dict

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find rows of given indexes
        regions_df_temp = (self.regions_df.iloc[indexes]).copy()
        regions_df_temp = regions_df_temp.reset_index(drop=True)

        # Generate data
        X, y = self.__data_generation(regions_df_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.regions_df))


        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, regions_df: pd.DataFrame):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        cool_generator = hic_loader(regions_df, self.cool_path, resolution=self.cool_resolution)
        chip_generator_dict = self.get_generator_dict(regions_df, self.chip_names, self.chip_path_dict)

        
        X = np.zeros((self.batch_size, *self.chip_dim))
        
        if self.upper_triangle:
            y = np.zeros((self.batch_size, (self.hic_dim[0]**2 + self.hic_dim[0])//2))
        else:
            y = np.zeros((self.batch_size, *self.hic_dim))
        

        # Generate data
        for i, row in regions_df.iterrows():
            # Store output
            mat = next(cool_generator)
            y[i] = mat[self.upper_indices]

            # Store input
            X[i] = np.array([next(chip_generator_dict[name]) for name in self.chip_names]).T

        return X, y