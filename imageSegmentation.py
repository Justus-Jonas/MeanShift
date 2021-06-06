import os

import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import argparse

from plotclusters3D import plotclusters3D


class Meanshift:
    def __init__(self, t, pos=False):
        self.radius = 0
        self.threshold = t
        self.pos = pos

    def findpeak2(self, data, idx, c):
        """
        finds the peak for a data point + tracks for optimization 2 --> for other implementation the
        return value is ignored --> extra computation very insignificant
        Args:
            data: data
            idx: index of poit
            c: c paramater

        Returns:

        """
        # for opt 2
        assigned_pixels = np.zeros(data.shape[0])

        center = data[idx]

        # set above threshold --> infinity
        shift_distance = np.math.inf
        while (self.threshold < shift_distance):
            # calculate distance
            points_distance = cdist([center],data)[0]
            # points in window
            points_in_window = data[points_distance < self.radius]
            # for opt2: add pixel along path
            assigned_pixels[np.where(points_distance < self.radius/c)] = 1
            # new center of mass
            new_center = points_in_window.mean(axis=0)
            # distance of shift --> for threshold
            shift_distance = np.linalg.norm(center-new_center)
            # new center
            center = new_center
        return center, assigned_pixels,points_distance


    def meanshift(self, data, r, c):
        labels = np.asarray([None] * data.shape[0])
        peaks_list = []

        pbar = tqdm.tqdm(total=data.shape[0], disable=False)

        for idx in range(0, data.shape[0]):

            peak, assigned_pixels, distances = self.findpeak2(data, idx, c)

            if not peaks_list:
                # set at the beginning so that we do not look for peaks (they can not exist yet)
                distances_peak = np.array(np.math.inf)
            else:
                distances_peak = cdist([peak], peaks_list)[0]

            # merge if 2 peaks too close
            if distances_peak.min() > r/2:
                peaks_list.append(peak)
                label = len(peaks_list) - 1
                labels[idx] = label
            else:
                labels[idx] = distances_peak.argmin()
            pbar.update(1)
        peaks_list = np.array(peaks_list)
        return peaks_list, labels


    def meanshift_opt1(self, data, r, c):
        labels = np.asarray([None] * data.shape[0])
        peaks_list = []

        pbar = tqdm.tqdm(total=data.shape[0], disable=False)

        for idx in range(0, data.shape[0]):
            # check if it was in radius of previous peak
            if labels[idx] is not None:
                continue
            peak, assigned_pixels, distances = self.findpeak2(data, idx, c)


            if not peaks_list:
                distances_peak = np.array(np.math.inf)
            else:
                distances_peak = cdist([peak], peaks_list)[0]

            # merge if 2 peaks too close
            if distances_peak.min() > r/2:
                peaks_list.append(peak)
                label = len(peaks_list) - 1
                labels[idx] = label
                # opt 1
                labels[distances < r] = label
            else:
                labels[idx] = distances_peak.argmin()
                # opt 1
                labels[distances < r] = distances_peak.argmin()
            pbar.update(1)
        peaks_list = np.array(peaks_list)
        return peaks_list, labels


    def meanshift_opt2(self, data, r, c):
        labels = np.asarray([None] * data.shape[0])
        peaks_list = []

        pbar = tqdm.tqdm(total=data.shape[0], disable=False)

        for idx in range(0, data.shape[0]):
            # check if it was in radius of previous peak
            if labels[idx] is not None:
                continue
            peak, assigned_pixels, distances = self.findpeak2(data, idx, c)

            if not peaks_list:
                distances_peak = np.array(np.math.inf)
            else:
                distances_peak = cdist([peak], peaks_list)[0]

            # merge if 2 peaks too close

            if distances_peak.min() > r/2:
                peaks_list.append(peak)
                label = len(peaks_list) - 1
                labels[idx] = label
                # opt 2
                labels[assigned_pixels == 1] = label
            else:
                labels[idx] = distances_peak.argmin()
                # opt 2
                labels[assigned_pixels == 1] = distances_peak.argmin()
            pbar.update(1)
        peaks_list = np.array(peaks_list)
        return peaks_list, labels


    def meanshift_opt(self, data, r, c):
        labels = np.asarray([None] * data.shape[0])
        peaks_list = []

        pbar = tqdm.tqdm(total=data.shape[0], disable=False)

        for idx in range(0, data.shape[0]):
            # check if it was in radius of previous peak
            if labels[idx] is not None:
                continue
            peak, assigned_pixels, distances = self.findpeak2(data, idx, c)


            if not peaks_list:
                distances_peak = np.array(np.math.inf)
            else:
                distances_peak = cdist([peak], peaks_list)[0]

            # merge if 2 peaks too close

            if distances_peak.min() > r/2:
                peaks_list.append(peak)
                label = len(peaks_list) - 1
                labels[idx] = label
                # opt 1
                labels[distances < r] = label
                # opt 2
                labels[assigned_pixels == 1] = label
            else:
                labels[idx] = distances_peak.argmin()
                # opt 1
                labels[distances < r] = label
                # opt 2
                labels[assigned_pixels == 1] = distances_peak.argmin()
            pbar.update(1)
        peaks_list = np.array(peaks_list)
        return peaks_list, labels


    def reconstruct_segmented_image(self,current_shape, final_shape, peaks, labels):
        """
        reconstructs image by assigning for every label the peak color
        Args:
            current_shape: shape of current array
            final_shape: shape of the image array
            peaks: list of all peaks
            labels: list of all labels

        Returns: reconstrcuted image (segmented image)

        """
        segmented_image = np.zeros(current_shape, dtype="uint8")
        for label in np.unique(labels):
            segmented_image[labels == label] = peaks[label][0:3]
        segmented_image = segmented_image.reshape(final_shape)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2BGR)
        return segmented_image


    def segment_image(self, image, r, c, algorithm):
        self.radius = r

        # change color space to CIELAB
        img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        print(img.shape)

        # reshape to 3 dimensional data points
        img_reshaped = img.reshape(-1, 3)

        # add positional information if required
        if self.pos:
            x, y = np.meshgrid(range(img.shape[0]), range(img.shape[1]))
            data = np.concatenate([img_reshaped, y.reshape(-1, 1), x.reshape(-1, 1)], axis=1)
        else:
            data = img_reshaped

        if algorithm == 0:
            peaks, labels = self.meanshift(data, r, c)
        elif algorithm == 1:
            peaks, labels = self.meanshift_opt1(data, r, c)
        elif algorithm == 2:
            peaks, labels = self.meanshift_opt2(data, r, c)
        elif algorithm == 3:
            peaks, labels = self.meanshift_opt(data, r, c)
        else:
            print('passed number for algorithm was invalid, using optimized mean shift')
            peaks, labels = self.meanshift_opt(data, r, c)

        # plot 3D Cluster deactivated for final version
        #plotclusters3D(data,labels,peaks)


        segmented_image = self.reconstruct_segmented_image(img_reshaped.shape,img.shape,peaks, labels)

        return segmented_image


def valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


def is_int(parser, arg):
    if type(arg) == int:
        return arg
    else:
        parser.error("argument has to be type int, %s is invalid!" % arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image segmentation")
    parser.add_argument("-f", dest="filename", required=True,
                        help="input file for segmentation", metavar="FILE",
                        type=lambda x: valid_file(parser, x))

    parser.add_argument("-r", dest="r", required=True,
                        help="input file with two matrices", metavar="Int",
                        type=lambda x: is_int(parser, int(x)))

    parser.add_argument("-c", dest="c", required=False,
                        help="takes int, if parsed Search Path Optimization is used", metavar="Int",
                        type=lambda x: is_int(parser, int(x)))

    parser.add_argument("-b", required=False,
                        help="if set, Basin  of attraction is used",action='store_true')

    parser.add_argument("-p", required=False,
                        help="if set, preprocessing with low pass filter is used",action='store_true')

    parser.add_argument("-pos", required=False,
                        help="if set, positional information is used",action='store_true')

    args = parser.parse_args()
    #image = cv2.imread("../data/exp2.jpg")
    image = cv2.imread(args.filename)
    print('image has been loaded')

    if args.pos:
        print('positional information used')
        mean_shift = Meanshift(0.01, True)
    else:
        print('no positional information used')
        mean_shift = Meanshift(0.01, False)

    if args.p:
        print('image has been blurred')
        kernel = np.ones((5, 5), np.float32) / 25
        image = cv2.filter2D(image, -1, kernel)

    if args.c:
        print('c paramter with valid value was received, using Search Path Optimization')
        segmented = mean_shift.segment_image(image,args.r,args.c,2)
    elif args.b:
        print('b paramter was received, using Barsin of attraction')
        segmented = mean_shift.segment_image(image, args.r, 4, 1)
    else:
        print('Neither a "b" nor a "c" parameter was received, using classic Meanshift')
        segmented = mean_shift.segment_image(image, args.r, 4, 0)
    plt.imshow(segmented)
    plt.savefig("segmentedImage.png")
    plt.show()
