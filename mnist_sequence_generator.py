#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
import argparse

import numpy as np


def load_data_and_dict():
    """Checks if numpy format images and idx dict exist
    if so:
      load
    if not:
      prepare and load

    :return: tuple of:
      <numpy array> mnist images
      <numpy array> dictionary
    """

    # check if data and dicts exist else download and generate
    mnist_dir = './data'
    mnist_images_fn = "./mnist_images"
    mnist_images_fn_loc = os.path.join(mnist_dir, mnist_images_fn + ".npy")
    mnist_idx_dict_fn = "./mnist_idx_dict"
    mnist_idx_dict_fn_loc = os.path.join(mnist_dir, mnist_idx_dict_fn + ".pickle")
    if os.path.isfile(mnist_images_fn_loc) and os.path.isfile(mnist_idx_dict_fn_loc):
        print("Found image data, loading...", end="")
        images = np.load(mnist_images_fn_loc)
        with open(mnist_idx_dict_fn_loc, 'rb') as handle:
            idx_dict = pickle.load(handle)
        print("DONE")
    else:
        print("Not all image data found, preparing...")
        import gzip
        import shutil
        import urllib.request
        mnist_urls = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                      "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"]

        # download if not exists
        os.makedirs(mnist_dir, exist_ok=True)

        # Download the file if it does not exist
        for mnist_url in mnist_urls:
            download_filename = os.path.join(mnist_dir, os.path.basename(mnist_url))
            if not os.path.isfile(download_filename):
                print(f"Downloading: {mnist_url}")
                urllib.request.urlretrieve(mnist_url, download_filename)

        # extract zip files if necessary
        zip_files = [os.path.join(mnist_dir, x) for x in os.listdir(mnist_dir) if x.endswith('.gz')]
        targets = [x[:-3] for x in zip_files]

        for idx, zip_file in enumerate(zip_files):
            if not os.path.isfile(targets[idx]):
                print(f"Unzipping {zip_file} to {targets[idx]}")
                with gzip.open(zip_file, 'rb') as f_in:
                    with open(targets[idx], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        from mnist import MNIST

        mndata = MNIST(mnist_dir)
        images, labels = mndata.load_training()
        images = np.array(images)
        labels = np.array(labels)

        idx_dict = {}
        for i in range(10):
            idx_dict[i] = np.where(labels == i)[0]

        # save data for future use
        print(f"Saving images to {mnist_images_fn_loc}...", end="")
        np.save(mnist_images_fn_loc[:-4], images)
        print("DONE")
        print(f"Saving idx dict to {mnist_idx_dict_fn_loc}...", end="")
        with open(mnist_idx_dict_fn_loc, 'wb') as handle:
            pickle.dump(idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("DONE")

    return images, idx_dict


def create_digit_sequence(n_arr, width, margin_min, margin_max, images, id_dict):
    """Creates a sequence of digits in a single image

    :return: 
      <numpy array> (height, width, 3) shaped image
    """
    if margin_max < margin_min:
        raise ValueError("Maximum margin must be larger or equal to minimum margin")

    image_size = 28
    res = np.zeros((image_size, width, 3))

    extra_margin = width - (len(n_arr) - 1) * margin_min - len(n_arr) * image_size

    if extra_margin < 0:
        raise ValueError("Current given minimum margin would result in exceeded width")

    start_idx = 0

    for x in n_arr:
        img = images[np.random.choice(id_dict[int(x)])].reshape((image_size, image_size, 1))
        res[:, start_idx:start_idx+image_size, :] = img
        start_idx += image_size
        additional_margin = np.random.randint(0, margin_max - margin_min)
        additional_margin = np.min((extra_margin, additional_margin))
        extra_margin -= additional_margin
        start_idx += additional_margin
    
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from MNIST images for OCR training purposes.')
    parser.add_argument('-w', '--width', default='200',
                        help='Width of the resulting image')
    parser.add_argument('-i', '--minmargin', default='0', type=int,
                        help='Minimum margin between MNIST characters')
    parser.add_argument('-a', '--maxmargin', default='100', type=int,
                        help='Maximum margin between MNIST characters')
    parser.add_argument('-l', '--strlen', default='5',
                        help='number of characters per string')
    parser.add_argument('-s', '--numberstring',
                        help='string of numbers ')
    parser.add_argument('-n', '--genn', default='10',
                        help='number of images to generate')
    parser.add_argument('-o', '--outputdir', default='./images',
                        help='output directory for generated images')
    args = parser.parse_args()

    # parse args to int if necessary
    mnist_string = args.numberstring
    min_margin = args.minmargin
    max_margin = args.maxmargin
    width = int(args.width)
    n = int(args.genn)
    str_len = int(args.strlen)
    char_string = args.numberstring

    # load images and idx data
    images, id_dict = load_data_and_dict()

    # create output dir if not exist
    out_dir = args.outputdir
    os.makedirs(out_dir, exist_ok=True)

    # main program loop
    for i in range(n):

        # generate a new string per loop if one wasn't provided
        if char_string is None:
            gen_arr = np.random.randint(0, 9, str_len)
        else:
            gen_arr = [int(x) for x in char_string]

        # run the generator
        mnist_ocr_image = create_digit_sequence(gen_arr, width, min_margin, max_margin, images, id_dict)
        np.save(os.path.join(out_dir, "mnist_ocr_image_{:0>6}".format(i)), mnist_ocr_image)

    print(f"Saved {n} images in {out_dir}")
