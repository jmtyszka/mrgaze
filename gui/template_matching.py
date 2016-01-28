import cv2.cv as cv
import cv2
import time
import numpy
import sys
import thread

# usage: 
# select ROI with mouse, followed by 'r' keyboard press
# select template region with mouse, follow by 't' keyboard press
# press 's' if you want to use sobel
# press 'm' to start template matching
# press 'esc' to quit

class ROI:

    def __init__(self, rect, frame):
        self.rect = list(rect)
        self.data = self.set_data(frame)
        self.data_orig = self.data.copy()
        self.sobel = False
        self.equalizeHist = False
        self.match_template = False
        self.get_brightness = False
        print "New ROI at %s" % self.rect

    def set_template(self, x, frame):
        self.template = frame[x[1]:x[3],x[0]:x[2],:].copy()
        print "Template set to %s" % x
        return

    def set_brightness(self, x):
        return

    def toggle_get_brightness(self):
        if self.get_brightness:
            self.get_brightness = False
        else:
            self.get_brightness = True
        return self.get_brightness
        
    def toggle_template_match(self):
        if self.match_template:
            self.match_template = False
        else:
            self.match_template = True
        return self.match_template

    def toggle_equalizeHist(self):
        if self.equalizeHist:
            self.equalizeHist = False
        else:
            self.equalizeHist = True
            print "equalizing hist"
        return self.equalizeHist

    def toggle_sobel(self):
        if self.sobel:
            self.sobel = False
        else:
            self.sobel = True
        return self.sobel

    def set_contrast(self, x):
        return
    
    def set_saturation(self, x):
        return

    def reset(self, frame):
        frame[self.rect[1]:self.rect[3],self.rect[0]:self.rect[2],:] = self.data_orig.copy()
        print "reset"
        return frame

#    def get_template_data(self, frame):
#        return frame[self.template[1]:self.template[3],self.template[0]:self.template[2],:].copy()

    def set_data(self, frame):
        return frame[self.rect[1]:self.rect[3],self.rect[0]:self.rect[2],:].copy()

class CallBack:

    def __init__(self):
        self.ROIs = list()
        self.current_frame_index = 1
        self.dont_update_after_setting_tb = True
        self.next_frame_index = 1
        self.last_trackbar_change = time.time()
        self.trackbar_thread_running = False
        self.playing_movie = False
        self.capture = cv2.VideoCapture(sys.argv[1])
        _, self.current_frame = self.capture.read()
        self.width, self.height = self.get_capture_dimensions(self.capture)
#        self.frames = numpy.empty((self.frame_cache_size, self.height, self.width, 3), dtype='uint8')
        self.total_frames = int(self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
#        self.frame_locations = [-1] * self.total_frames
        self.fps = int(self.capture.get(cv.CV_CAP_PROP_FPS))
 #        self.init_video_cache()
#        self.caching = False
        cv2.namedWindow("Camera", cv2.CV_WINDOW_AUTOSIZE );
        cv2.moveWindow("Camera", 10,20)
#        self.template = self.current_frame.copy()
#        self.imageRoi = [0,0,self.width,self.height]
        self.selection = [0,0,self.width,self.height]
        cv2.createTrackbar("frame", "Camera", self.current_frame_index, self.total_frames, self.on_trackbar)
#        cv2.createTrackbar("nFrames", "Camera", self.nSelectedFrames, self.nFrames, self.on_trackbar_nFrames)
        self.on_trackbar(self.current_frame_index)
#        self.on_trackbar_nFrames(self.current_frame_index)
        param = 0
        cv2.setMouseCallback("Camera",self.on_mouse, param)

        self.with_sobel = False
        self.with_temp_match = False
        self.changed_frame = False
        self.overlayText = "Start by selecting region with mouse."

#        self.text_font = cv2.initFont(cv2.CV_FONT_HERSHEY_COMPLEX, .5, .5, 0.0, 1, cv.CV_AA )
#        self.text_coord = ( 5, 15 )
#        self.text_color = cv2.CV_RGB(255,100,150)

    def on_trackbar(self, index):
        if self.dont_update_after_setting_tb:
#            self.current_frame_index = self.current_frame_index - 1
            self.dont_update_after_setting_tb = False
            return
        self.next_frame_index = index
        self.last_trackbar_change = time.time()
        if self.trackbar_thread_running == False:
            thread.start_new_thread(self.get_frame, ("",))


    def on_mouse(self,event, x, y, flag, param):

        if(event == cv2.EVENT_LBUTTONDOWN):
           self.selection[0] = x
           self.selection[1] = y
#           print "mouse down"
        elif (event == cv2.EVENT_LBUTTONUP):
           self.selection[2] = x
           self.selection[3] = y
           my_frame = self.current_frame.copy()
           cv2.rectangle(my_frame, tuple(self.selection[0:2]), tuple(self.selection[2:4]), cv.CV_RGB(200,120,120), 1)
 #          self.setOverlay('selected')
           cv2.imshow("Camera", my_frame)
    
    def get_capture_dimensions(self, capture):
        """Get the dimensions of a capture"""
        width = int(capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        return width, height

    def get_frame(self, dummy):
        # text_color = (250,250,250) #color as (B,G,R)
        # cv2.putText(self.current_frame, "Seeking ...", (10,10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=text_color, thickness=1, lineType=cv2.CV_AA)
        # cv2.imshow("Camera", self.current_frame)
        # cv2.waitKey(10)
        resume_playing_after = False
        self.trackbar_thread_running = True
        if self.playing_movie:
            self.playing_movie = False
            resume_playing_after = True
        while time.time() - self.last_trackbar_change < .5:
            time.sleep(.01)
        print "Target frame: %s " % self.next_frame_index
        print "Current frame: %s" % self.current_frame_index
        if self.next_frame_index < self.current_frame_index:
            self.capture = cv2.VideoCapture(sys.argv[1])
            x = 0
        else:
            x = self.current_frame_index
        while x < self.next_frame_index:
            frame = self.capture.grab()
            x = x + 1
        print "Arrive at frame: %s" % x
        _, frame = self.capture.read()
        self.current_frame = frame
        # cv2.imshow("Camera", self.current_frame)
        # cv2.waitKey(10)
        self.current_frame_index = x + 1
        self.trackbar_thread_running = False
        if resume_playing_after:
            self.playing_movie = True


    def run_equalize(self, img):
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)
        img = cv2.cvtColor( img, cv2.COLOR_GRAY2RGB )
        return img

    def run_sobel(self, img):
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        img = cv2.GaussianBlur( img, (9,9), 0 )
        img_x = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        img_y = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        img_x = cv2.convertScaleAbs(img_x)
        img_y = cv2.convertScaleAbs(img_y)
        img = cv2.addWeighted(img_x, .5, img_y, .5, 0)
        img = cv2.cvtColor( img, cv2.COLOR_GRAY2RGB )
        return img

    def calc_brightness(self, img):
        tmp = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        return cv2.mean(tmp)
   
    def match_template(self, roi):
        roi_img = roi.data
#        roi_img = roi.get_data(self.current_frame)
        result = cv2.matchTemplate(roi_img, roi.template, cv.CV_TM_SQDIFF)
        minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(result)
        matchLoc = (minLoc[0] + roi.rect[0], minLoc[1] + roi.rect[1])
        W,H = roi.template[:,:,1].shape
        if True:
            cv2.rectangle(self.current_frame, matchLoc, ( matchLoc[0] + H , matchLoc[1] + W ), cv.CV_RGB(120,200,120), 1)
            cv2.rectangle(self.current_frame, (roi.rect[0],roi.rect[1]), (roi.rect[2],roi.rect[3]), cv.CV_RGB(200,120,120), 1)
        return matchLoc

    # def setOverlay(self,event):
    #     if event == "reset":
    #         self.overlayText = "Start by selecting region with mouse."
    #     elif event == 'selected':
    #         self.overlayText = "Press R to use selection as region of interest,\nPress T to use selection as region of interest."

    # def showOverlay(self):
    #     x = 10
    #     y = 20
    #     text_color = (250,250,250) #color as (B,G,R)
    #     cv2.putText(self.current_frame, self.overlayText, (x,y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=text_color, thickness=1, lineType=cv2.CV_AA)

    def delete_roi(self, roi):
        self.ROIs[roi].reset(self.current_frame)
        new_ROIs = list()
        for i in range(0, len(self.ROIs)):
            if i != roi:
                new_ROIs.append(self.ROIs[i])
        self.ROIs = new_ROIs
        self.current_roi = len(self.ROIs) -1

    def proc_rois(self,log=False):
        for roi in self.ROIs:
            roi.data = roi.set_data(self.current_frame)
        for roi in self.ROIs:
            if roi.sobel:
                roi.data = self.run_sobel(roi.data)
                self.current_frame[roi.rect[1]:roi.rect[3],roi.rect[0]:roi.rect[2]] = roi.data
            if roi.equalizeHist:
                roi.data = self.run_equalize(roi.data)
                self.current_frame[roi.rect[1]:roi.rect[3],roi.rect[0]:roi.rect[2]] = roi.data
            if roi.match_template:
#                print "Doing template matching"
                roi.match_loc = self.match_template(roi)
            if roi.get_brightness:
                roi.brightness = self.calc_brightness(roi.data)
        if log:
            outstring = ""
            for roi in self.ROIs:
                try:
#                    print roi.brightness
                    if roi.get_brightness:
                        outstring += "%s \t" % roi.brightness[0]
                    if roi.match_template:
                        outstring += "%s \t %s \t" % (roi.match_loc[0], roi.match_loc[1])
                except AttributeError as inst:
                    print type(inst)     # the exception instance
                    print inst.args      # arguments stored in .args
                    print inst 
                    return 
            outstring = outstring.strip() + "\n"
            return outstring

    def process_all(self):
        filename = sys.argv[1][0:len(sys.argv[1])-4] + ".log"
        print "name of log file: %s" % filename
        outfile = file(filename, 'w')
        self.capture = cv2.VideoCapture(sys.argv[1])
        self.current_frame_index = 0
        outstring = ""
        x = 0
        for roi in self.ROIs:
            if roi.get_brightness:
                outstring += "avg\t"
            if roi.match_template:
                outstring += "x%s\ty%s\t" % (x, x)
            x += 1
        outstring = [outstring.strip() + "\n"]
#        outstring = outstring.strip() + '\n'
        print self.total_frames/100
        while self.current_frame_index < self.total_frames:
            _, self.current_frame = self.capture.read()
            outstring.append(self.proc_rois(True))
#            print outstring
            self.current_frame_index = self.current_frame_index + 1
            self.dont_update_after_setting_tb = True
            if self.current_frame_index % (self.total_frames/1000) == 0:
                outfile.writelines(outstring)
                outstring = []
                cv2.imshow("Camera", self.current_frame)
                cv2.setTrackbarPos("frame", "Camera", self.current_frame_index)
                filename = "snapshots/" + sys.argv[1][0:len(sys.argv[1])-4] + ("_%06d" % self.current_frame_index) + ".jpg"
#                print filename
                cv2.imwrite(filename,self.current_frame)
                c = cv2.waitKey(1)
                if c == 13: # esc (END)
                    return False
#        print sys.argv[1]
#       print sys.argv[1].split(".")
        outfile.close()
        return True

    def save_snapshot(self):
        filename = sys.argv[1][0:len(sys.argv[1])-4] + ".jpg"
        cv2.imwrite(filename,self.current_frame)

    def show_roi_numbers(self):
        index = 0
        for roi in self.ROIs:
            x = roi.rect[0]
            y = roi.rect[1]
            text_color = (250,250,250) #color as (B,G,R)
            cv2.putText(self.current_frame, "%s" % index, (x,y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=text_color, thickness=1, lineType=cv2.CV_AA)
            cv2.imshow("Camera", self.current_frame)
            c = cv2.waitKey(10)
            index = index + 1

    def config(self):
        c = -1
        previous_frame = self.current_frame.copy()
        current_frame_index = self.current_frame_index
#        self.showOverlay()
        cv2.imshow("Camera", self.current_frame)
        while True: # not enter or esc
            self.changed_frame = False
            c = cv2.waitKey(1000/self.fps)
            # c = cv2.waitKey(1000)
            if c > 1000000:
                c = c - 1048576
            if c > -1:
                print c
            if self.playing_movie:
                # if self.current_frame_index == self.total_frames - 1:
                #     self.playing_movie = False
                #     continue
                _, self.current_frame = self.capture.read()
                self.current_frame_index = self.current_frame_index + 1
                self.dont_update_after_setting_tb = True
                cv2.setTrackbarPos("frame", "Camera", self.current_frame_index)
            if c == 13: # 
                self.process_all()
            if c == 27 or c == 1048603: # esc (quit)
                print "Exiting"
                return
            if c == 32: # space
                if self.playing_movie == False:
                    self.playing_movie = True
                else:
                    self.playing_movie = False
            if c == 46: # /
                self.show_roi_numbers()
            if c == 47: # .
                self.save_snapshot()
            if c == 97: # a
                self.ROIs[self.current_roi].toggle_get_brightness()
            if c == 101: # e
                self.ROIs[self.current_roi].toggle_equalizeHist()
                self.run_equalize(self.ROIs[self.current_roi].data)
                self.changed_frame = True
            if c == 109: # m
                self.ROIs[self.current_roi].toggle_template_match()
            if c == 114: # r
#                self.imageRoi = list(self.selection)
                self.ROIs.append(ROI(self.selection, self.current_frame))
                self.current_roi = len(self.ROIs) - 1
            if c == 115: #s
                self.ROIs[self.current_roi].toggle_sobel()
#                self.run_sobel(self.ROIs[self.current_roi].data)
                self.changed_frame = True
#                print "do sobel"
            if c == 116: # t
                self.ROIs[self.current_roi].set_template(self.selection, self.current_frame)
            if c == 117: # u
                self.ROIs[self.current_roi].reset(self.current_frame)
                self.changed_frame = True
            if c == 100: # d
                self.delete_roi(self.current_roi)
            if c in range(48,58):
                if c - 48 in range(0,len(self.ROIs)):
                    self.current_roi = c - 48
                else:
                    print "don't have that roi yet"
            if self.changed_frame or self.current_frame_index != current_frame_index:
                self.proc_rois()
                cv2.imshow("Camera", self.current_frame)
                current_frame_index = self.current_frame_index
            




if __name__ == '__main__':
    cb = CallBack()
    cb.config()
