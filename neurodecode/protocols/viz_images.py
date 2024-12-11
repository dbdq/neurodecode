from __future__ import print_function, division

"""
Bar visual feedback class


Kyuhwa Lee, 2015
Swiss Federal Institute of Technology (EPFL)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import os
import sys
import cv2
import time
import gzip
import numpy as np
import neurodecode.utils.q_common as qc
from neurodecode import logger
try:
    import cPickle as pickle  # Python 2 (cPickle = C version of pickle)
except ImportError:
    import pickle  # Python 3 (C version is the default)


def combine_images(image_paths, labels, pickle_file, resize=None):
    """
    Save images as a single pickle file for faster loading

    TODO: Generalise directions to argument-based names

    Input
    -----
    image_paths: list of directories for each label
    labels: list of labels that correspond to image_paths
    pickle_file: combined output file
    resize: [width, height] or None for original size
    """
    tm = qc.Timer()
    images = {}
    for i, label in enumerate(labels):
        logger.info('[%s] Reading images from %s' % (label, image_paths[i]))
        images[label] = read_images(image_paths[i], resize)
        if len(images[label]) == 0:
            logger.warn('No images were loaded for %s' % label)
    qc.save_obj(pickle_file, images)
    ''' compressing and decompressing turns out to be too slow
    logger.info('Merging and compressing ...')
    g = gzip.compress(pickle.dumps(images))
    with gzip.open(pickle_file, "wb") as f:
        f.write(g)
    '''
    logger.info('Took %.1f s' % tm.sec())


def read_images(img_path, screen_size=None):
    pnglist = []
    for f in qc.get_file_list(img_path):
        if f[-4:] != '.png':
            continue

        img = cv2.imread(f)
        # fit to screen size if image is larger
        if screen_size is not None:
            screen_width, screen_height = screen_size
            rx = img.shape[1] / screen_width
            ry = img.shape[0] / screen_height
            if max(rx, ry) > 1:
                if rx > ry:
                    target_w = screen_width
                    target_h = int(img.shape[0] / rx)
                elif rx < ry:
                    target_w = int(img.shape[1] / ry)
                    target_h = screen_height
                else:
                    target_w = screen_width
                    target_h = screen_height
            else:
                target_w = img.shape[1]
                target_h = img.shape[0]
            dsize = (int(target_w), int(target_h))
            img_res = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
            img_out = np.zeros((screen_height, screen_width, img.shape[2]), dtype=img.dtype)
            ox = int((screen_width - target_w) / 2)
            oy = int((screen_height - target_h) / 2)
            img_out[oy:oy+target_h, ox:ox+target_w, :] = img_res
        else:
            img_out = img
        pnglist.append(img_out)
        print('.', end='')
    print()
    return pnglist


class ImageVisual(object):
    # Default setting
    colors = dict(G=(20, 140, 0), B=(255, 90, 0), R=(0, 50, 200), Y=(0, 215, 235),
        K=(0, 0, 0), W=(255, 255, 255), w=(200, 200, 200))
    barwidth = 100
    window_name = 'Protocol'

    def __init__(self, image_object, show_feedback=True, screen_pos=None, screen_size=None):
        """
        Input:
            image_object: pickle file generated using combine_images()
            show_feedback: show the feedback on the screen?
            screen_pos: screen position in (x,y)
            screen_size: screen size in (w,h)
        """
        # screen size and message setting
        if screen_size is None:
            if sys.platform.startswith('win'):
                from win32api import GetSystemMetrics
                screen_width = GetSystemMetrics(0)
                screen_height = GetSystemMetrics(1)
            else:
                screen_width = 1920
                screen_height = 1080
            screen_size = (screen_width, screen_height)
        else:
            screen_width, screen_height = screen_size
        if screen_pos is None:
            screen_x, screen_y = (0, 0)
        else:
            screen_x, screen_y = screen_pos

        self.img = np.zeros((screen_height, screen_width, 3), np.uint8)
        self.img_black = np.zeros((screen_height, screen_width, 3), np.uint8)
        self.img_white = self.img_black + 255
        self.set_show_feedback(show_feedback)
        self.set_cue_color(boxcol='B', crosscol='W')
        self.width = self.img_black.shape[1]
        self.height = self.img_black.shape[0]
        hw = int(self.barwidth / 2)
        self.cx = int(self.width / 2)
        self.cy = int(self.height / 2)
        self.xl1 = self.cx - hw
        self.xl2 = self.xl1 - self.barwidth
        self.xr1 = self.cx + hw
        self.xr2 = self.xr1 + self.barwidth
        self.yl1 = self.cy - hw
        self.yl2 = self.yl1 - self.barwidth
        self.yr1 = self.cy + hw
        self.yr2 = self.yr1 + self.barwidth

        # load pickled images
        # note: this is painfully slow in Pytohn 2 even with cPickle (3s vs 27s)
        assert image_object[-4:] == '.pkl', 'Check if the file is in Python Pickle format'
        logger.info('Loading image binary file %s ...' % image_object)
        tm = qc.Timer()
        self.images = qc.load_obj(image_object)
        image_shape = list(self.images.values())[0][0].shape # [height, width]
        feedback_w = image_shape[1] / 2
        feedback_h = image_shape[0] / 2
        loc_x = [int(self.cx - feedback_w), int(self.cx + feedback_w)]
        loc_y = [int(self.cy - feedback_h), int(self.cy + feedback_h)]
        img_fit = np.zeros((screen_height, screen_width, 3), np.uint8)

        # adjust to the current screen size
        if image_shape[0] != screen_height or image_shape[1] != screen_width:
            logger.info('Fitting images to the screen size')
            for label, image_batch in self.images.items():
                for i, img in enumerate(image_batch):
                    img_fit = np.zeros((screen_height, screen_width, 3), np.uint8)
                    img_fit[loc_y[0]:loc_y[1], loc_x[0]:loc_x[1]] = img
                    self.images[label][i] = img_fit

        logger.info('Loading took %.1f s.' % tm.sec())
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(self.window_name, screen_x, screen_y)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
        self.blank()

    def finish(self):
        cv2.destroyAllWindows()

    def set_show_feedback(self, fb):
        self.show_feedback = fb

    """ reserved for bar feedback """
    def set_cue_color(self, boxcol='B', crosscol='W'):
        self.boxcol = self.colors[boxcol]
        self.crosscol = self.colors[crosscol]

    """ blank image """
    def blank(self):
        self.img.fill(0)

    """ fill image with specific color """
    def fill(self, color):
        self.img = np.full(self.img.shape, self.colors[color], dtype=np.uint8)

    """ draw cue with custom colors """
    def draw_cue(self, label):
        self.img = self.images[label][0]

    """ show image that corresponds to the decoder probability (or score) """
    def move(self, label, dx, caption='', caption_color='W'):
        if label not in self.images:
            logger.error('Undefined label %s' % label)
        if self.show_feedback:
            self.img = self.images[label][dx]
        if len(caption) > 0:
            self.put_text(caption, color=caption_color)

    """ embed texts to the current image """
    def put_text(self, txt, color='W', x=None, y=None, scale=1, thickness=1):
        self.img = self.img.copy()
        size_wh, baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        if x is None:
            x = int(self.cx - size_wh[0] / 2)
        if y is None:
            y = int(self.cy - size_wh[1] / 2)
        pos = (x, y)
        cv2.putText(self.img, txt, pos, cv2.FONT_HERSHEY_DUPLEX, scale, self.colors[color], thickness, cv2.LINE_AA)

    """ Update the graphics and return any captured key strokes """
    def update(self):
        cv2.imshow(self.window_name, self.img)
        return cv2.waitKey(1)
