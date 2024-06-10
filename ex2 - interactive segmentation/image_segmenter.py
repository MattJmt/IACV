from os import WCONTINUED
from typing_extensions import ChainMap
import numpy as np

from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx
)

from extract_patches import extract_patches
from advanced_methods import perform_min_cut


class ImageSegmenter:
    def __init__(self, k_fg=2, k_bg=5, mode='kmeans'):
        """ Feel free to add any hyper-parameters to the ImageSegmenter.
            
            But note:
            For the final submission the default hyper-parameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        
        # Number of clusters in FG/BG
        self.k_fg = k_fg
        self.k_bg = k_bg
        
        self.mode= mode
        
    def extract_features_(self, sample_dd):
        """ Extract features, e.g. p x p neighborhood of pixel, from the RGB image """
        
        img = sample_dd['img']
        H, W, C = img.shape
        
        #
        # TO IMPLEMENT
        #
        # xy layers
        x = np.tile(np.arange(W).reshape(1, W, 1),(H, 1, 1))
        y = np.tile(np.arange(H).reshape(H, 1, 1),(1, W, 1))

        img = np.concatenate((img, x, y), axis=2)
        img = extract_patches(img, 1)
        img = img.reshape(H*W, C+2)
        return img
    
    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        features = self.extract_features_(sample_dd)
        
        #
        # TO IMPLEMENT
        #

        mask_fg = sample_dd['scribble_fg'].reshape(H*W, 1)
        mask_bg = sample_dd['scribble_bg'].reshape(H*W, 1)
        img_fg = features[mask_fg.flatten().astype(bool)]
        img_bg = features[mask_bg.flatten().astype(bool)]
        centroids_fg = kmeans_fit(img_fg, self.k_fg)
        centroids_bg = kmeans_fit(img_bg, self.k_bg)

        n = H*W
        im_mask = np.zeros((n,1))
        for i in range(n):
          norms_fg = np.sort(np.linalg.norm(centroids_fg - features[i], axis=1))
          norms_bg = np.sort(np.linalg.norm(centroids_bg - features[i], axis=1))
          if norms_fg[0] <= norms_bg[0]:
            im_mask[i] = 1
        im_mask = im_mask.reshape(H,W)
        
        return im_mask

    def segment_image_grabcut(self, sample_dd):
        """ Segment via an energy minimisation """

        # Foreground potential set to 1 inside box, 0 otherwise
        unary_fg = sample_dd['scribble_fg'].astype(np.float32) / 255

        # Background potential set to 0 inside box, 1 everywhere else
        unary_bg = 1 - unary_fg

        # Pairwise potential set to 1 everywhere
        pairwise = np.ones_like(unary_fg)

        # Perfirm min cut to get segmentation mask
        im_mask = perform_min_cut(unary_fg, unary_bg, pairwise)
        
        return im_mask

    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)
        
        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)
        
        elif self.mode == 'grabcut':
            return self.segment_image_grabcut(sample_dd)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")