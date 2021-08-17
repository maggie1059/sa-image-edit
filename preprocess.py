import numpy as np
import tensorflow as tf
import os

# Returns an iterator into loaded training dataset
def load_image_batch(dir_name, batch_size=1, shuffle_buffer_size=250000, n_threads=2):
    # Function used to load and pre-process image files
    def load_and_process_image(file_path):
        # Load image
        image = tf.io.decode_png(tf.io.read_file(file_path), channels=3)

        # Convert image to normalized float (0, 1)
        c0 = tf.image.convert_image_dtype(image, tf.float32)
        src = tf.slice(c0, [0, 0, 0], [256, 256, 3])
        
        c = tf.slice(c0, [0, 256, 0], [256, 256, 2])
        c0 = tf.slice(c, [0, 0, 0], [256, 256, 1])
        c1 = tf.slice(c, [0, 0, 1], [256, 256, 1])
        # Create C' as mask of known texels in C
        c_prime = tf.zeros([256, 256, 1])
        c_prime = tf.where(c0 > 0, tf.ones([256, 256, 1]), c_prime)
        c_prime = tf.where(c1 > 0, tf.ones([256, 256, 1]), c_prime)
        
        # Rescale data to range (-1, 1)
        c = (c - 0.5) * 2
        x = tf.range(256, dtype=tf.float32)
        y = tf.range(256, dtype=tf.float32)
        X, Y = tf.meshgrid(x, y)
        X = tf.expand_dims(X, axis=-1)
        Y = tf.expand_dims(Y, axis=-1)
        mg = tf.concat([X, Y], axis=-1)
        mg = mg/255.0 #normalize meshgrid
        new = tf.concat([src, c, c_prime, mg], axis=-1)
        return new

    # This assumes that the directory has subfolders for each texture in the dataset used
    ds = None
    for root, dirs, files in os.walk(dir_name):
        for dirname in sorted(dirs, key=int):
            # List file names/file paths
            dir_path = dir_name + '/' + dirname + '/*.png'
            dataset = tf.data.Dataset.list_files(dir_path)

            # Load and process images in parallel
            dataset = dataset.map(map_func=load_and_process_image, num_parallel_calls=n_threads)
            
            # Create batch and drop any incomplete batches
            dataset = dataset.batch(batch_size, drop_remainder=True)

            # Concatenate datasets from subfolders
            if ds==None:
                ds = dataset
            else:
                ds = ds.concatenate(dataset)
    # Return an iterator over this dataset
    ds = ds.prefetch(1)
    return ds

# Returns an iterator into loaded test dataset
def load_test_batch(dir_name, batch_size=1, shuffle_buffer_size=250000, n_threads=2):
    def load_and_process_image(file_path):
        # Load image
        image = tf.io.decode_png(tf.io.read_file(file_path), channels=3)
        # Convert image to normalized float (0, 1)
        c = tf.image.convert_image_dtype(image, tf.float32)
        return c
    # Traverse subdirectories for each texture (sorted by texture so that training/testing
    # images can be matched up correctly by texture)
    ds = None
    for root, dirs, files in os.walk(dir_name):
        for dirname in sorted(dirs, key=int):
            dir_path = dir_name + '/' + dirname + '/*.png'
            dataset = tf.data.Dataset.list_files(dir_path)
            dataset = dataset.map(map_func=load_and_process_image, num_parallel_calls=n_threads)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            if ds==None:
                ds = dataset
            else:
                ds = ds.concatenate(dataset)

    ds = ds.prefetch(1)
    return ds