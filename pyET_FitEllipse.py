#!/opt/local/bin/python
#
# Video pupilometry functions
# - takes calibration and gaze video filenames as input
# - controls calibration and gaze estimation workflow
#
# USAGE : pyET.py <Calibration Video> <Gaze Video>
#
# AUTHOR : Mike Tyszka
# PLACE  : Caltech
# DATES  : 2014-05-07 JMT From scratch
#
# This file is part of pyET.
#
#    pyET is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    pyET is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#   along with pyET.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2014 California Institute of Technology.

import numpy as np
import random
import cv2

#---------------------------------------------
# RANSACE Ellipse Fitting Functions
#---------------------------------------------

def FitEllipse_RANSAC(pnts, gray):
    
    # Graphic display flag
    do_graphic = True
    
    # Maximum normalized error squared for inliers
    max_norm_err_sq = 4.0
    
    # Init best ellipse and support
    best_ellipse = ((0,0),(1,1),0)
    best_support = -np.inf    
    
    # Create display window and init overlay image
    if do_graphic:
        cv2.namedWindow('RANSAC', cv2.WINDOW_NORMAL)
        overlay = cv2.cvtColor(gray/2,cv2.COLOR_GRAY2RGB)
    
    # Count pnts (n x 2)
    n_pnts = pnts.shape[0]
    
    # Break if too few points to fit ellipse (RARE)
    if n_pnts < 5:
        return best_ellipse
    
    # Ransac iterations
    for itt in range(0,3):
        
        # Select 5 points at random
        pnts_random5 = np.asarray(random.sample(pnts, 5))

        # Fit ellipse to points        
        ellipse = cv2.fitEllipse(pnts_random5)

        # Calculate normalized errors for all points
        norm_err = EllipseNormError(pnts, ellipse)
        
        # Identify inliers (normalized error < 2.0)
        # np.nonzero returns a tuple wrapped array - take first element
        inliers = np.nonzero(norm_err**2 < max_norm_err_sq)[0]
        
        print('Itt %d Init  : Found %d inliers' % (itt, inliers.size))
        
        # Extract inlier points
        pnts_inliers = pnts[inliers]
        
        # Fit ellipse to inlier set
        ellipse_inliers = cv2.fitEllipse(pnts_inliers)
        
        # Update overlay image and display
        if do_graphic:
            OverlayRANSACFit(overlay, pnts, pnts_inliers, ellipse_inliers)
            cv2.imshow('RANSAC', overlay)
            cv2.waitKey(5)
        
        # Refine inliers iteratively
        for refine in range(0,2):
            
            # Recalculate normalized errors for the inliers
            norm_err_inliers = EllipseNormError(pnts_inliers, ellipse_inliers)
            
            # Identify inliers
            inliers = np.nonzero(norm_err_inliers**2 < max_norm_err_sq)[0]
            
            print('Itt %d Ref %d : Found %d inliers' % (itt, refine, inliers.size))
            
            # Update inliers set
            pnts_inliers = pnts_inliers[inliers]
            
            # Fit ellipse to refined inlier set
            ellipse_inliers = cv2.fitEllipse(pnts_inliers)

            # Update overlay image and display
            if do_graphic:
                OverlayRANSACFit(overlay, pnts, pnts_inliers, ellipse_inliers)
                cv2.imshow('RANSAC', overlay)
                cv2.waitKey(5)

        # Calculate support for the refined inliers
        support = EllipseSupport(inliers, norm_err_inliers)

        # Count inliers (n x 2)
        n_inliers    = inliers.size
        perc_inliers = (n_inliers * 100.0) / n_pnts 
        
        # Update best ellipse
        if support > best_support:
            best_support = support
            best_ellipse = ellipse_inliers
            best_pnts = pnts_inliers

        # Report on this iteration
        print("RANSAC Iteration   : %d" % itt)
        print("  Support (Best)   : %0.1f %0.1f" % (support, best_support))
        print("  Inliers          : %d / %d (%0.1f%%)" % (n_inliers, n_pnts, perc_inliers))
    
    # RANSAC finished
    print('RANSAC Complete')    
    
    # Final best result
    if do_graphic:
        OverlayRANSACFit(overlay, pnts, best_pnts, best_ellipse)
        cv2.imshow('RANSAC', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return best_ellipse


def EllipseError(pnts, ellipse):
 
    # Calculate algebraic distances and gradients of all points from fitted ellipse
    distance, grad, absgrad, normgrad = ConicFunctions(pnts, ellipse)

    # Calculate error from distance and gradient
    # See Swirski et al 2012
    # TODO : May have to use distance / |grad|^0.45 - see Swirski code

    # Gradient array has x and y components in rows (see ConicFunctions)
    err = distance / absgrad
    
    return err


def EllipseNormError(pnts, ellipse):
    
    # Error normalization factor, alpha
    # Normalizes cost to 1.0 at point 1 pixel out from minor vertex along minor axis

    # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
    # Where aa and bb are the major and minor axes, and phi_b_deg
    # is the CW x to minor axis rotation in degrees
    (x0,y0), (bb,aa), phi_b_deg = ellipse
    
    # Semiminor axis
    b = bb/2

    # Convert phi_b from deg to rad
    phi_b_rad = phi_b_deg * np.pi / 180.0
    
    # Minor axis vector
    bx, by = np.cos(phi_b_rad), np.sin(phi_b_rad)
    
    # Point one pixel out from ellipse on minor axis
    p1 = np.array( (x0 + (b + 1) * bx, y0 + (b + 1) * by) ).reshape(1,2)

    # Error at this point
    err_p1 = EllipseError(p1, ellipse)
    
    print('Normalizing error : %0.3f' % err_p1)
    
    # Errors at provided points
    err_pnts = EllipseError(pnts, ellipse)
    
    return err_pnts / err_p1


def EllipseSupport(inliers, norm_err_inliers):
    
    # Reciprocal RMS error of inlier points
    support = 1.0 / np.sqrt(np.mean(norm_err_inliers[inliers]**2))
    
    return support
    
    
#---------------------------------------------
# Ellipse Math
#---------------------------------------------

#
# Geometric to conic parameter conversion
# Adapted from Swirski's ConicSection.h
# https://bitbucket.org/Leszek/pupil-tracker/
#
def Geometric2Conic(ellipse):
    
    # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
    # Where aa and bb are the major and minor axes, and phi_b_deg
    # is the CW x to minor axis rotation in degrees
    (x0,y0), (bb,aa), phi_b_deg = ellipse
    
    # Semimajor and semiminor axes
    a, b = aa/2, bb/2

    # Convert phi_b from deg to rad
    phi_b_rad = phi_b_deg * np.pi / 180.0
    
    # Major axis unit vector
    ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)
    
    # Useful intermediates
    a2 = a*a
    b2 = b*b
    
    A = ax*ax / a2 + ay*ay / b2;
    B = 2*ax*ay / a2 - 2*ax*ay / b2;
    C = ay*ay / a2 + ax*ax / b2;
    D = (-2*ax*ay*y0 - 2*ax*ax*x0) / a2 + (2*ax*ay*y0 - 2*ay*ay*x0) / b2;
    E = (-2*ax*ay*x0 - 2*ay*ay*y0) / a2 + (2*ax*ay*x0 - 2*ax*ax*y0) / b2;
    F = (2*ax*ay*x0*y0 + ax*ax*x0*x0 + ay*ay*y0*y0) / a2 + (-2*ax*ay*x0*y0 + ay*ay*x0*x0 + ax*ax*y0*y0) / b2 - 1;

    # Compose conic parameter array
    conic = np.array((A,B,C,D,E,F))

    return conic


#
# Merge geometric parameter functions from van Foreest code
# http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
#
def Conic2Geometric(conic):
    
    # Extract modified conic parameters    
    A,B,C,D,E,F = conic[0], conic[1]/2, conic[2], conic[3]/2, conic[4]/2, conic[5]
    
    # Usefult intermediates
    dAC = A-C
    Z = np.sqrt( 1 + 4*B*B/(dAC*dAC) )
    
    # Center
    num = B * B - A * C
    x0 = (C * D - B * E) / num
    y0 = (A * E - B * D) / num

    # Axis lengths
    up    = 2 * (A*E*E + C*D*D + F*B*B - 2*B*D*E - A*C*F)
    down1 = (B*B-A*C) * ( -dAC*Z - (C+A) )
    down2 = (B*B-A*C) * (  dAC*Z - (C+A) )
    b, a  = np.sqrt(up/down1), np.sqrt(up/down2)   
    
    # Minor axis rotation angle in degrees (CW from x axis, origin upper left)
    phi_b_deg =  0.5 * np.arctan(2 * B / dAC) * 180.0 / np.pi
    
    # Note OpenCV ellipse parameter format
    return (x0,y0), (b,a), phi_b_deg
    
#
# Conic quadratic curve support functions
# Adapted from Swirski's ConicSection.h
# https://bitbucket.org/Leszek/pupil-tracker/
#
    
def ConicFunctions(pnts, ellipse):
    
    # General 2D quadratic curve (biquadratic)
    # Q = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
    # For point on ellipse, Q = 0, with appropriate coefficients 
    
    # Convert from geometric to conic ellipse parameters
    conic = Geometric2Conic(ellipse)

    # Row vector of conic parameters (Axx, Axy, Ayy, Ax, Ay, A1) (1 x 6)
    C = np.array(conic)
    
    # Extract vectors of x and y values
    x, y = pnts[:,0], pnts[:,1]
    
    # Construct polynomial array (6 x n)
    X = np.array( ( x*x, x*y, y*y, x, y, np.ones_like(x) ) )
    
    # Calculate Q/distance for all points (1 x n)
    distance = C.dot(X)
    
    # Quadratic curve gradient at (x,y)
    # Analytical grad of Q = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
    # (dQ/dx, dQ/dy) = (2Ax + By + D, Bx + 2Cy + E)
    
    # Construct conic gradient coefficients vector (2 x 3)
    Cg = np.array( ( (2*C[0], C[1], C[3]), (C[1], 2*C[2], C[4]) ) )    
    
    # Construct polynomial array (3 x n)
    Xg = np.array( (x, y, np.ones_like(x) ) )
    
    # Gradient array (2 x n)
    grad = Cg.dot(Xg)
    
    # Normalize gradient -> unit gradient vector
    absgrad = np.apply_along_axis(np.linalg.norm, 0, grad)
    normgrad = grad / absgrad
    
    return distance, grad, absgrad, normgrad

#
# Graphics functions
#

def OverlayRANSACFit(img, all_pnts, inlier_pnts, ellipse):

    # NOTE : all points are (x,y) pairs, but arrays are (row, col)
    # so swap coordinate ordering for correct positioning in array

    # Overlay all pnts in red
    for col,row in all_pnts:
        img[row,col] = [0,0,255]
    
    # Overlay inliers in green
    for col,row in inlier_pnts:
        img[row,col] = [0,255,0]
 
    # Overlay inlier fitted ellipse in yellow
    cv2.ellipse(img, ellipse, (0,255,255), 1)