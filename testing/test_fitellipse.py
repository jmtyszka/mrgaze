#!/opt/local/bin/python

import cv2
import numpy as np
import ConfigParser
from mrgaze import pupilometry, media, config

def main():
    
    # Setup default config structure
    print('Initializing configuration')
    cfg = ConfigParser.ConfigParser()
    cfg = config.InitConfig(cfg)
    
    # Update defaults
    cfg.set('VIDEO','downsampling','1')
    cfg.set('PUPILSEG','method','otsu')
    cfg.set('PUPILSEG','thresholdperc','50.0')
    cfg.set('PUPILSEG','pupildiameterperc','15.0')
    cfg.set('PUPILSEG','sigma','0.0')
    cfg.set('PUPILFIT','method','ROBUST_LSQ')
    cfg.set('PUPILFIT','maxrefinements','5')

    # Load test eye tracking frame
    print('Loading test frame')
    test_frame = '/Users/jmt/GitHub/mrgaze/testing/CBIC_Example_2.png'
    frame = media.load_image(test_frame, cfg)
    
    # Init ROI to whole frame
    # Note (col, row) = (x, y) for shape
    x0, x1, y0, y1 = 0, frame.shape[1], 0, frame.shape[0]

    # Define ROI rect
    roi_rect = (x0,y0),(x1,y1)
        
    # Extract pupil ROI (note row,col indexing of image array)
    roi = frame[y0:y1,x0:x1]
    
    # Find glint(s) in frame
    glints, glints_mask, roi_noglints = pupilometry.FindGlints(roi, cfg)
    
    # Segment pupil intelligently - also return glint mask
    print('Segmenting pupil')
    pupil_bw, roi_rescaled = pupilometry.SegmentPupil(roi, cfg)
    
    # Create composite image of various stages of segmentation
    strip_bw = np.hstack((roi, pupil_bw * 255, glints_mask * 255, roi_rescaled))
    
    # Init montage
    montage_rgb = np.array([])
    
    # Fit ellipse to pupil boundary - returns ellipse ROI
    for method in ('RANSAC_SUPPORT','RANSAC','ROBUST_LSQ','LSQ'):
        
        print('Fitting pupil ellipse : %s' % method)

        cfg.set('PUPILFIT','method',method)

        eroi = pupilometry.FitPupil(pupil_bw, roi, cfg)
            
        # Construct pupil ellipse tuple
        pupil_ellipse = (eroi[0][0], eroi[0][1]), eroi[1], eroi[2]

        # TODO: find best glint candidate in glint mask
        glint = pupilometry.FindBestGlint(glints_mask, pupil_ellipse)
            
        # RGB version of preprocessed frame for output video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
        # Create RGB overlay of pupilometry on ROI
        frame_rgb = pupilometry.OverlayPupil(frame_rgb, pupil_ellipse, roi_rect, glint)
        
        if montage_rgb.size == 0:
            montage_rgb = frame_rgb
        else:
            montage_rgb = np.hstack((montage_rgb, frame_rgb))
    
    cv2.imshow('Segmentation', strip_bw)
    cv2.imshow('Pupilometry', montage_rgb)
    cv2.waitKey()
    
    print('Done')
    

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()

