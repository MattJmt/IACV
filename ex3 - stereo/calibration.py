import numpy as np
from zmq import ZMQBindError


def compute_mx_my(calib_dict):
    """
    Given a calibration dictionary, compute mx and my (in units of [px/mm]).
    
    mx -> Number of pixels per millimeter in x direction (ie width)
    my -> Number of pixels per millimeter in y direction (ie height)
    """
    
    #
    # TO IMPLEMENT
    #
    my = calib_dict["height"] /calib_dict["aperture_h"]
    mx = calib_dict["width"] /calib_dict["aperture_w"]

    return mx, my


def estimate_f_b(calib_dict, calib_points, n_points=None):
    """
    Estimate focal lenght f and baseline b from provided calibration points.

    Note:
    In real life multiple points are useful for calibration - in case there are erroneous points.
    Here, this is not the case. It's OK to use a single point to estimate f, b.
    
    Args:
        calib_dict (dict)           ... Incomplete calibaration dictionary
        calib_points (pd.DataFrame) ... Calibration points provided with data. (Units are given in [mm])
        n_points (int)              ... Number of points used for estimation
        
    Returns:
        f   ... Focal lenght [mm]
        b   ... Baseline [mm]
    """
    if n_points is not None:
        calib_points = calib_points.head(n_points)
    else: 
        n_points = len(calib_points)

    #
    # TO IMPLEMENT
    #
    """
    o_x = calib_dict["o_x"]
    o_y = calib_dict["o_y"]
    mx = calib_dict["mx"]
    my = calib_dict["my"]
    ul = calib_points["ul [px]"].values
    vl = calib_points["vl [px]"].values
    ur = calib_points["ur [px]"].values
    vr = calib_points["vr [px]"].values
    X = calib_points["X [mm]"].values
    Y = calib_points["Y [mm]"].values
    Z = calib_points["Z [mm]"].values

    
    # Calculate fxl, fyl, fxr, fyr
    fxl = Z * (ul - o_x) / (mx * X)
    fyl = Z * (vl - o_y) / (my * Y)
    fxr = Z * (ur - o_x) / (mx * X)
    fyr = Z * (vr - o_y) / (my * Y)
    
    f = np.round(np.median(np.concatenate([fxl, fyl, fxr, fyr])))
    
    b_sum = np.sum(X - (ur - o_x) * Z / (f * mx))
    b = np.round(b_sum / len(calib_points))
    return f, b
    """
    mx, my = compute_mx_my(calib_dict)
    f = round(((calib_points["ul [px]"][0]-calib_dict["o_x"])*calib_points["Z [mm]"][0]/(calib_points["X [mm]"][0]*mx)),5)
    b = round(calib_points["X [mm]"][0] - (calib_points["ur [px]"][0]-calib_dict["o_x"])*calib_points["Z [mm]"][0]/(f*mx),5)
    return f,b