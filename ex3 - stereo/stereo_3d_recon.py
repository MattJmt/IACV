import numpy as np

from calibration import compute_mx_my, estimate_f_b
from extract_patches import extract_patches


def triangulate(u_left, u_right, v, calib_dict):
    """
    Triangulate (determine 3D world coordinates) a set of points given their projected coordinates in two images.
    These equations are according to the simple setup, where C' = (b, 0, 0)

    Args:
        u_left  (np.array of shape (num_points,))   ... Projected u-coordinates of the 3D-points in the left image
        u_right (np.array of shape (num_points,))   ... Projected u-coordinates of the 3D-points in the right image
        v       (np.array of shape (num_points,))   ... Projected v-coordinates of the 3D-points (same for both images)
        calib_dict (dict)                           ... Dict containing camera parameters
    
    Returns:
        points (np.array of shape (num_points, 3)   ... Triangulated 3D coordinates of the input - in units of [mm]
    """
    #
    # TO IMPLEMENT
    mx = calib_dict["mx"]
    my = calib_dict["my"]
    o_x = calib_dict["o_x"]
    o_y = calib_dict["o_y"]
    f = calib_dict["f"]
    b = calib_dict["b"]

    denominator = (u_left - u_right)
    zero_mask = (denominator != 0)  # Mask to handle division by zero

    X = b * (u_left - o_x) / denominator
    Y = b * mx / my * (v - o_y) / denominator
    Z = b * mx * f / denominator

    coords = np.array([X,Y,Z])  # Assuming x, y, z are already arrays
    points = np.hstack(coords.transpose())

    return points


def compute_ncc(img_l, img_r, p):
    """
    Calculate normalized cross-correlation (NCC) between patches at the same row in two images.
    
    The regions near the boundary of the image, where the patches go out of image, are ignored.
    That is, for an input image, "p" number of rows and columns will be ignored on each side.

    For input images of size (H, W, C), the output will be an array of size (H - 2*p, W - 2*p, W - 2*p)

    Args:
        img_l (np.array of shape (H, W, C)) ... Left image
        img_r (np.array of shape (H, W, C)) ... Right image
        p (int):                            ... Defines square neighborhood. Patch-size is (2*p+1, 2*p+1).
                              
    Returns:
        corr    ... (np.array of shape (H - 2*p, W - 2*p, W - 2*p))
                    The value output[r, c_l, c_r] denotes the NCC between the patch centered at (r + p, c_l + p) 
                    in the left image and the patch centered at  (r + p, c_r + p) at the right image.
    """

    # Add dummy channel dimension
    if img_l.ndim == 2:
        img_l = img_l[:, :, None]
        img_r = img_r[:, :, None]
    
    assert img_l.ndim == 3, f"Expected 3 dimensional input. Got {img_l.shape}"
    assert img_l.shape == img_r.shape, "Shape mismatch."
    
    H, W, C = img_l.shape

    patch_size = (2*p+1)**2

    intensity_r = np.sum(img_r, axis=2)/3
    intensity_l = np.sum(img_l, axis=2)/3

    patches_l = extract_patches(img_l, 2*p+1)
    patches_r = extract_patches(img_r, 2*p+1)
    
    i = 0
    for y in range(-p, p+1):
        for x in range(-p, p+1):
            patches_l[:, :, i] = np.roll(np.roll(intensity_l, -y, axis=0), -x, axis=1)
            patches_r[:, :, i] = np.roll(np.roll(intensity_r, -y, axis=0), -x, axis=1)
            i += 1

    # Standardize each patch
    patches_l_mean = patches_l.mean(axis=2, keepdims=True)
    patches_r_mean = patches_r.mean(axis=2, keepdims=True)

    patches_l_std = np.sqrt(((patches_l - patches_l_mean) ** 2).mean(axis=2, keepdims=True))
    patches_r_std = np.sqrt(((patches_r - patches_r_mean) ** 2).mean(axis=2, keepdims=True))

    # Avoid division by zero
    patches_l_norm = np.divide((patches_l - patches_l_mean), patches_l_std, out=np.zeros_like((patches_l - patches_l_mean)), where=patches_l_std != 0)
    patches_r_norm = np.divide((patches_r - patches_r_mean), patches_r_std, out=np.zeros_like((patches_r - patches_r_mean)), where=patches_r_std != 0)

    
    # Compute correlation (using matrix multiplication) - corr will be of shape H, W, W
    corr = np.matmul(patches_l_norm, np.transpose(patches_r_norm, (0, 2, 1))) / patch_size
    corr *= (patches_l_std > 0)

    return corr[p:H-p, p:W-p, p:W-p]


class Stereo3dReconstructor:
    def __init__(self, p=7, w_mode='none'):
        self.p = p
        self.w_mode = w_mode

    def fill_calib_dict(self, calib_dict, calib_points):
        calib_dict['mx'], calib_dict['my'] = compute_mx_my(calib_dict)
        calib_dict['f'], calib_dict['b'] = estimate_f_b(calib_dict, calib_points)
        return calib_dict

    def recon_scene_3d(self, img_l, img_r, calib_dict):
        if img_l.ndim == 2:
            img_l = img_l[:, :, None]
            img_r = img_r[:, :, None]

        assert img_l.ndim == 3, f"Expected 3-dimensional input. Got {img_l.shape}"
        assert img_l.shape == img_r.shape, "Shape mismatch."

        H, W, C = img_l.shape

        H_small, W_small = H - 2 * self.p, W - 2 * self.p

        calib_small = calib_dict
        calib_small['height'], calib_small['width'] = H_small, W_small
        calib_small['o_x'] = calib_dict['o_x'] - self.p
        calib_small['o_y'] = calib_dict['o_y'] - self.p

        y, u_left = np.meshgrid(
            np.arange(H_small, dtype=float),
            np.arange(W_small, dtype=float),
            indexing='ij'
        )

        # Compute normalized cross correlation & find correspondence
        corr = compute_ncc(img_l, img_r, self.p)
        corr = np.tril(corr, k=-1)

        # Find correspondence
        u_right = np.argmax(corr, axis=2)

        # Set certainty
        if self.w_mode == 'none':
            certainty_score = np.ones((H_small, W_small), dtype=float)
        else:
            raise NotImplementedError("Implement your own certainty estimation")

        v = y + self.p  # Adjust v coordinate

        points = triangulate(u_left.flatten(), u_right.flatten(), v.flatten(), calib_small)
        points = points.reshape(H_small, W_small, 3)

        # certainty scores
        v_max = np.max(corr, axis=2)
        delta = np.max(v_max) - np.min(v_max)
        certainty_score = (v_max - np.min(v_max)) / delta
        certainty_score = np.pad(certainty_score, self.p)[:, :, None]

        # Pad the results
        points = np.pad(points, ((self.p, self.p), (self.p, self.p), (0, 0)))

        return np.concatenate([points, certainty_score], axis=2)
