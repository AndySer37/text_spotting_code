#! /usr/bin/env python
import rospy
import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from os.path import expanduser
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.msg import Rect, Rects

from mvnc import mvncapi as mvnc
home = expanduser("~")

import time
class NCS_node():
    def __init__(self):
        self.switch_img = 0
        self.initial()
        #self.camera_name = rospy.get_param('~camera_name')
        self.image_sub = rospy.Subscriber("/kara/camera_node/image/compressed", CompressedImage, self.img_cb, queue_size=1)
        #self.quad_sub = rospy.Subscriber("/"+self.camera_name+"/quad_proposals", Rects, self.img_crop)
        #self.quad_sub = rospy.Subscriber("/atlas/quad_proposals", Rects, self.img_crop)
        self.image_pub = rospy.Publisher('gray', Image, queue_size=10)
        self.bridge = CvBridge()
        self.cv_image = 0
        self.img_region = 0
        self.cv_img_crop = []
        self.switch_quad = 0
        #self.switch_img = 0
        #NCS params
        #self.model = model = 'street_en_harvest'
        #self.start = 0
        #self.time = 0
        #self.n = 1

    def initial(self):
        self.model = model = 'street_en_harvest'
        self.start = 0
        self.time = 0
        self.n = 1

        self.camera_name = rospy.get_param('~camera_name')
        self.device_work = False
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
        self.deviceCheck()
        self.dim = (100, 32) #(width, height)

    def deviceCheck(self):
        #check device is plugged in
        self.devices = mvnc.EnumerateDevices()
        if len(self.devices) == 0:
            self.device_work = False
            rospy.loginfo('NCS device not found')
	else:
            self.device_work = True
            rospy.loginfo('NCS device found')
            self.initialDevice()

    def initialDevice(self):
        # set the blob, label and graph
        self.device = mvnc.Device(self.devices[0])
        self.device.OpenDevice()
        #network_blob = home + "/" + self.model + '.graph'
        network_blob = home + "/" + "88200_prune_0"
        #Load blob
        with open(network_blob, mode='rb') as f:
            blob = f.read()

        self.graph = self.device.AllocateGraph(blob)
        self.switch_img = 1
        print "open"

    def img_cb(self, data):
        #print "Image callback"
        self.switch_img += 1
        if self.switch_img != 60:
            #print "wait for prediction"
            return
        #self.switch_img = 0
        try:
            self.switch_img = 0
            #print "switch_off"
            self.start = data.header.stamp.secs
            start = time.time()
            #self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            np_arr = np.fromstring(data.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img_gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            mser = cv2.MSER_create(12, 100, 250000, 0.25, 0.2, 200, 1.01, 0.003, 5)
            regions, _ = mser.detectRegions(img_gray)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
            #cv2.polylines(gray_img, hulls, 1, (255, 0, 0), 2)
            imgContours = self.cv_image
            contour_list = []
            for i, contour in enumerate(hulls):
                x,y,w,h = cv2.boundingRect(contour)
                #repeat = self.checkContourRepeat(contour_list, x, y, w, h)
                #img_region = img_cv[y:y+h, x:x+w]      
                if 2*h < w and h*w < 10000 and h*w > 1000:
                    cv2.rectangle(imgContours,(x, y),(x+w, y+h),(0,255,0),3)
                    img_region = self.cv_image[y:y+h, x:x+w]
                    self.cv_img_crop.append(img_region)
            image_all_mser_image = Image()
            image_all_mser_image.header = rospy.Time.now
            image_all_mser_image = self.bridge.cv2_to_imgmsg(imgContours, "bgr8")
            self.image_pub.publish(image_all_mser_image)
            print "detection time:",time.time()-start
            self.ncs()      
            
        except CvBridgeError as e:
            print(e)

    def ncs(self):
        print "receive proposal"
        i = 0
        for im in self.cv_img_crop:
        #im = self.cv_image
            start = time.time()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            if im is None:
                break
            im = cv2.resize(im, self.dim)         
            im = im.astype(np.float32)
            #im = im/255.0
            #im -= np.mean(im)
            #im = im.astype(np.float32)

            # Send the image to NCS
            self.graph.LoadTensor(im.astype(np.float16), 'user object')
            
            output, userobj = self.graph.GetResult()
            if i == 0:
                now = rospy.get_rostime().secs
                self.time += (now-self.start)
                #print self.n, self.time
                self.n += 1
            i += 1 
            #order = output.argsort()[::-1][:4]
            top1 = output.argmax()

            #if output[top1] >= 0.9:
            print 'class: ',top1
            print output[top1] 
            print "prediction time:", time.time()-start

        self.cv_img_crop = []
        #self.switch_img = 1
        #print "switch_on"

    def onShutdown(self):
        rospy.loginfo("[%s] Shutdown." %(self.node_name))
    
def main(args):
    rospy.init_node('ncs_node', anonymous = False)
    ic = NCS_node()
    rospy.on_shutdown(ic.onShutdown)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
