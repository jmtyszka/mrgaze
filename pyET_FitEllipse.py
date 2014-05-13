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
    
    # Init best ellipse and support
    best_ellipse = ((0,0),(1,1),0)
    best_support = -np.inf    
    
    # Create display window and init overlay image
    if do_graphic:
        cv2.namedWindow('RANSAC', cv2.WINDOW_AUTOSIZE)
        overlay = cv2.cvtColor(gray/2,cv2.COLOR_GRAY2RGB)
    
    # Count pnts (n x 2)
    n_pnts = pnts.shape[0]
    
    # Ransac iterations
    for itt in range(0,2):
        
        # Select 5 points at random
        pnts_random5 = np.asarray(random.sample(pnts, 5))

        # Fit ellipse to points        
        ellipse = cv2.fitEllipse(pnts_random5)

        # Calculate normalized errors for all points
        norm_err = EllipseNormError(pnts, ellipse)
        
        # Identify inliers (normalized error < 2.0)
        inliers = np.array(np.nonzero(norm_err < 2.0))
        n_inliers = inliers.shape[1]
        
        print('Found %d inliers' % n_inliers)
        
        # Exctract inlier points
        pnts_inliers = pnts[inliers]
        
        # Fit ellipse to inlier set
        ellipse_inliers = cv2.fitEllipse(pnts_inliers)
        
        # Update overlay image and display
        if do_graphic:
            overlay = OverlayEllipse(overlay, pnts_inliers, ellipse_inliers)
            cv2.imshow('RANSAC', overlay)
            cv2.waitKey(5)
        
        # Refine inliers iteratively
        for refine in range(0,2):
            
            # Recalculate normalized errors for the inliers
            norm_err_inliers = EllipseNormError(pnts_inliers, ellipse_inliers)
            
            # Identify inliers
            inliers = np.nonzero(norm_err_inliers < 2.0)
            
            # Update inliers set
            pnts_inliers = pnts_inliers[inliers]
            
            # Fit ellipse to refined inlier set
            ellipse_inliers = cv2.fitEllipse(pnts_inliers)

            # Update overlay image and display
            if do_graphic:
                overlay = OverlayEllipse(overlay, pnts_inliers, ellipse_inliers)
                cv2.imshow('RANSAC', overlay)
                cv2.waitKey(5)
            
        # Calculate support for the refined inliers
        support = EllipseSupport(inliers, norm_err_inliers)

        # Count inliers (n x 2)
        n_inliers    = inliers.shape[0]
        perc_inliers = n_inliers / n_pnts * 100.0 

        # Report on this iteration
        print("RANSAC Iteration   : %d" % itt)
        print("  Support (Best)   : %0.1f %0.1f" % (support, best_support))
        print("  Inliers          : %d / %d (%0.1f%%)" % (n_inliers, n_pnts, perc_inliers))
        
        # Update best ellipse
        if support > best_support:
            best_support = support
            best_ellipse = ellipse_inliers
    
    return best_ellipse


def EllipseError(pnts, ellipse):
 
    # Calculate algebraic distances and gradients of all points from fitted ellipse
    distance, grad, absgrad, normgrad = ConicFunctions(pnts, ellipse)

    # Calculate error from distance and gradient
    # See Swirski et al 2012
    # TODO : May have to use distance / |grad|^0.45 - see Swirski code
    err = distance / absgrad   
    
    return err


def EllipseNormError(pnts, ellipse):
    
    # Error normalization factor, alpha
    # Normalizes cost to 1.0 at point 1 pixel out from minor vertex along minor axis

    # Ellipse tuple has form ( ( x0, y0), (a, b), phi_deg) )
    (x0,y0), (a,b), phi_deg = ellipse

    # Convert phi from deg to rad
    phi_rad = phi_deg * np.pi / 180.0
    
    # Minor axis vector
    bx, by = np.sin(phi_rad), np.cos(phi_rad)
    
    # Point 1 pixel out from ellipse on minor axis
    p1 = np.array( (x0 + (b + 1) * bx, y0 + (b + 1) * by) )

    # Error at this point
    err_p1 = EllipseError(p1, ellipse)
    
    # Errors at provided points
    err_pnts = EllipseError(pnts, ellipse)
    
    return err_pnts / err_p1


def EllipseSupport(inliers, norm_err_inliers, ):
    
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
    
    # Ellipse tuple has form ( ( x0, y0), (a, b), phi_deg) )
    (x0,y0), (a,b), phi_deg = ellipse

    # Convert phi from deg to rad
    phi_rad = phi_deg * np.pi / 180.0
    
    # Major axis unit vector
    ax, ay = np.cos(phi_rad), np.sin(phi_rad)
    
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
    a,b,c,d,f,g = conic[0], conic[1]/2, conic[2], conic[3]/2, conic[4]/2, conic[5]
    
    # Center
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    center = np.array((x0, y0))

    # Axis lengths
    up    = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1 = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2 = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    axes  = np.array( ( np.sqrt(up/down1), np.sqrt(up/down2) ) )
    
    # Rotation angle
    angle =  0.5 * np.arctan(2 * b / (a - c))    
    
    return center, axes, angle
    
#
# Conic quadratic curve support functions
# Adapted from Swirski's ConicSection.h
# https://bitbucket.org/Leszek/pupil-tracker/
#
    
def ConicFunctions(pnts, ellipse):
    
    # Convert from geometric to conic ellipse parameters
    conic = Geometric2Conic(ellipse)

    # Row vector of conic parameters (Axx, Axy, Ayy, Ax, Ay, A1) (1 x 6)
    C = np.array(conic)
    
    # Extract vectors of x and y values
    # x and y are in columns, so transpose pnts
    x, y = pnts.T
    
    # Construct polynomial array (6 x n)
    X = np.array( ( x*x, x*y, y*y, x, y, np.ones_like(x) ) )
    
    # Calculate vector of distances (1 x n)
    distance = C.dot(X)
    
    # Quadratic curve gradient at (x,y)
    # (Gx, Gy) = (2Ax + By + D, Bx + 2Cy + E)
    
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

def OverlayEllipse(overlay, pnts, ellipse):

    # Overlay all pnts in red
    for p in pnts:
        overlay[p[0],p[1]] = [0,0,255]
    
    # Overlay inliers in green
 
    # Overlay inlier fitted ellipse in yellow
    cv2.ellipse(overlay, ellipse, (0,255,0), 1)
    
    return overlay