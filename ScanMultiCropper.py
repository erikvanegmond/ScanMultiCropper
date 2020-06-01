import os
import cv2
import piexif
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import scipy.cluster.hierarchy as hcluster


class ScanMultiCropper:

    def __init__(self, scan_dir="", output_dir="", detector="gauss", **kwargs):
        self.scan_dir = scan_dir
        self.output_dir = output_dir
        self.detector = detector
        print(f"Processing files in {self.scan_dir} and moving cropped photos to {self.output_dir}")

    def run(self, save=True, show=False, photo=None, crop=True):
        for filename in os.listdir(self.scan_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")) and (not photo or filename == photo):
                self._process_file(filename, save=save, show=show, crop=crop)

    def _process_file(self, filename, save, show, crop):
        print(filename)
        scale_factor = 0.4
        ori_img = cv2.imread(os.path.join(self.scan_dir, filename))
        img = cv2.resize(ori_img, None, fx=scale_factor, fy=scale_factor)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        boundaries = self.find_boundaries(gray)
        if show:
            print(f"found {len(boundaries)} photos")
            self.draw_boundaries(img, boundaries, color=True)
        if not crop:
            return
        pil_im = Image.open(os.path.join(self.scan_dir, filename))

        if save:
            self._crop(filename, boundaries, pil_im, scale_factor)

    def find_boundaries(self, gray_img, ):
        scale_factor = 0.03

        if self.detector == "blobs":
            blobbed_img = self._simple_blobs(gray_img)
            data, clusters = self._find_clusters(blobbed_img, scale_factor)
        elif self.detector == "edges":
            blobbed_img = self._edge_based_blobs(gray_img)
            data, clusters = self._find_clusters(blobbed_img, scale_factor)
        elif self.detector == "gauss":
            blobbed_img = self._adaptive_gaussian_blobs(gray_img)
            data, clusters = self._find_clusters(blobbed_img, scale_factor)
        else:
            raise ValueError(f"Unknown detector type {self.detector}, choose from blobs, edges, gauss")

        initial_boundaries = self._find_large_image_bounding_boxes(data, clusters, scale_factor)
        boundaries = self._find_precise_boundaries(initial_boundaries, blobbed_img)
        return boundaries

    @staticmethod
    def _find_precise_boundaries(initial_boundaries, blobbed_img):
        boundaries = []
        # TODO what happens when not the entire photo is black.
        for mins, maxs, means in initial_boundaries:
            h, w = blobbed_img.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            flood_fill = blobbed_img.copy()
            cv2.floodFill(flood_fill, mask, tuple(reversed(list(map(lambda m: int(m), means)))), 100)
            filled = np.argwhere(flood_fill == 100)
            [x, y, w, h] = cv2.boundingRect(filled)
            boundaries.append([(x, y), (x + w, y + h), means])
        return boundaries

    def draw_boundaries(self, img, boundaries, color=False):
        line_image = img.copy()
        for mins, maxs, means in boundaries:
            cv2.rectangle(line_image, (mins[1], mins[0]), (maxs[1], maxs[0]), (0, 255, 0), 5)
        self.plot(line_image, color=color)

    @staticmethod
    def plot(img, title="", fig_size=None, color=False):
        if not fig_size:
            fig_size = (10, 10)
        plt.figure(figsize=fig_size)
        plt.title(title)
        if color:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_img)
        else:
            plt.imshow(img, cmap="gray")
        plt.show()

    @staticmethod
    def _find_large_image_bounding_boxes(data, clusters, scale_factor):
        initial_boundaries = []
        cluster_counts = Counter(clusters)
        for cluster in range(1, len(set(clusters)) + 1):
            if cluster_counts[cluster] > 300:
                mins = list(map(lambda x: max(x, 0),
                                (
                                        (
                                                data[np.where(clusters == cluster)].min(axis=0) - 2
                                        ) / scale_factor
                                ).astype(int).tolist()
                                ))
                maxs = ((data[np.where(clusters == cluster)].max(axis=0) + 2) / scale_factor).astype(int).tolist()
                means = data[np.where(clusters == cluster)].mean(axis=0) / scale_factor
                initial_boundaries.append([mins, maxs, means])
        return initial_boundaries

    @staticmethod
    def _find_clusters(blobbed_img, scale_factor):
        small_blobbed = cv2.resize(blobbed_img, None, fx=scale_factor, fy=scale_factor)
        data = np.argwhere(small_blobbed == 0)

        cluster_thresh = 1.1
        clusters = hcluster.fclusterdata(data, cluster_thresh, criterion="distance")
        return data, clusters

    ##########################
    # Methods to find photos #
    ##########################

    @staticmethod
    def _simple_blobs(gray_img):
        kernel_size = 15
        border_size = 15
        blur_gray = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

        ret, thresh = cv2.threshold(blur_gray, 238, 255, cv2.THRESH_BINARY)

        bordered = cv2.copyMakeBorder(thresh, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255))

        for _ in range(2):
            bordered = cv2.GaussianBlur(bordered, (kernel_size, kernel_size), 0)
            ret, bordered = cv2.threshold(bordered, 100, 255, cv2.THRESH_BINARY)

        y_size, x_size = bordered.shape
        no_borders = bordered[border_size:y_size - border_size, border_size:x_size - border_size]
        return no_borders

    def _edge_based_blobs(self, gray_img):
        low_threshold = 5  # 10
        high_threshold = 40  # 70
        edges = cv2.Canny(gray_img, low_threshold, high_threshold)
        edges = cv2.bitwise_not(edges)
        kernel_size = 5
        for i in range(8):
            edges = cv2.GaussianBlur(edges, (kernel_size, kernel_size), 0)
            ret, edges = cv2.threshold(edges, 180, 255, cv2.THRESH_BINARY)

            if i < 2:
                contours, hierarchy = cv2.findContours(cv2.bitwise_not(edges), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                short_contours = [contour for contour in contours if cv2.contourArea(contour) < 200]
                cv2.fillPoly(edges, pts=short_contours, color=(255, 255, 255))

        filled = self._fill_holes(edges)
        return filled

    def _adaptive_gaussian_blobs(self, gray_img):
        blobs = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        kernel = np.ones((7, 7), np.uint8)
        opening = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(cv2.bitwise_not(opening), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        short_contours = [contour for contour in contours if cv2.contourArea(contour) < 200]
        cv2.fillPoly(opening, pts=short_contours, color=(255, 255, 255))

        opening = self._fill_holes(opening)
        kernel_size = 11
        for i in range(2):
            opening = cv2.GaussianBlur(opening, (kernel_size, kernel_size), 0)
            ret, opening = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)

        filled = self._fill_holes(opening)
        return filled

    @staticmethod
    def _fill_holes(img):
        # add border to ensure we can flood from the outside
        border_size = 15
        bordered = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255))
        bordered = cv2.bitwise_not(bordered)
        im_flood_fill = bordered.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = bordered.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Flood fill from point (0, 0)
        cv2.floodFill(im_flood_fill, mask, (0, 0), 255)

        # Invert flood filled image
        im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)

        # Combine the two images to get the foreground.
        im_out = cv2.bitwise_not(bordered | im_flood_fill_inv)
        y_size, x_size = im_out.shape
        no_borders = im_out[border_size:y_size - border_size, border_size:x_size - border_size]
        return no_borders

    @staticmethod
    def _get_exif(img):
        exif_dict = piexif.load(img.info["exif"])
        exif_dict.pop('thumbnail', None)
        return exif_dict

    def _save(self, file_path, img):
        exif_dict = self._get_exif(img)
        img.save(os.path.join(self.output_dir, file_path), "JPEG", exif=piexif.dump(exif_dict))

    def _crop(self, filename, boundaries, pil_im, scale_factor):
        for i, [mins, maxs, _] in enumerate(boundaries):
            crop = pil_im.crop(
                (
                    int(mins[1] / scale_factor),
                    int(mins[0] / scale_factor),
                    int(maxs[1] / scale_factor),
                    int(maxs[0] / scale_factor)
                )
            )
            self._save(filename.rsplit(".")[0] + f"_{i + 1}" + ".jpg", crop)


class DatedScanMultiCropper(ScanMultiCropper):
    def __init__(self, year, month=1, day=1, **kwargs):
        print(kwargs)
        super().__init__(**kwargs)
        self.year = year
        self.month = month
        self.day = day
        print(f"Saved photos will have the date {year}-{month}-{day}")

    def _get_exif(self, img):
        exif_dict = super()._get_exif(img)
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = f"{self.year}:{self.month}:{self.day}"
        return exif_dict


class TaggedScanMultiCropper(ScanMultiCropper):
    def __init__(self, tags, **kwargs):
        print(kwargs)
        super().__init__(**kwargs)
        self.tags = tags
        print(f"Saved photos will have tags: {tags}")

    def _get_exif(self, img):
        exif_dict = super()._get_exif(img)
        ucs2 = []
        for c in self.tags:
            ucs2 += [ord(c), 0]
        exif_dict["0th"][piexif.ImageIFD.XPKeywords] = ucs2
        return exif_dict
