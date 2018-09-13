#! /usr/bin/env python
import rospy
import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image
from os.path import expanduser
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.msg import Rect, Rects

from mvnc import mvncapi as mvnc
home = expanduser("~")

import time
class NCS_node():
    def __init__(self):
        self.initial()
        #self.camera_name = rospy.get_param('~camera_name')
        self.image_sub = rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.img_cb)
        #self.quad_sub = rospy.Subscriber("/"+self.camera_name+"/quad_proposals", Rects, self.img_crop)
        self.quad_sub = rospy.Subscriber("/atlas/quad_proposals", Rects, self.img_crop)
        self.image_pub = rospy.Publisher('gray', Image, queue_size=10)
        self.bridge = CvBridge()
        self.cv_image = 0
        self.cv_img_crop = []
        self.switch_quad = 0
        self.switch_img = 1
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
        network_blob = home + "/" + self.model + '.graph'

        #Load blob
        with open(network_blob, mode='rb') as f:
            blob = f.read()

        self.graph = self.device.AllocateGraph(blob)

    def img_cb(self, data):
        #print "Image callback"
        if self.switch_img is 0:
            return
        try:
            self.start = data.header.stamp.secs
            #print self.start
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.switch_quad = 1
            self.switch_img = 0

        except CvBridgeError as e:
            print(e)

    def img_crop(self, quads):
        #print "Quad callback"
        if self.switch_quad is 0:
            return
        for quad in quads.rects:
            #print quad
            img = self.cv_image[quad.y:quad.y+quad.h, quad.x:quad.x+quad.w]
            self.cv_img_crop.append(img)
        self.switch_quad = 0

    def ncs(self):
        if type(self.cv_image) == np.int:
            print "No image receive."
            return

        if self.switch_img is not 0 or self.switch_quad is not 0:
            return
        i = 0
        for im in self.cv_img_crop:
        #im = self.cv_image
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

            if output[top1] >= 0.9:
                print 'class: ',top1
                print output[top1] 

        self.cv_img_crop = []
        self.switch_img = 1
    
def main(args):
    rospy.init_node('NCS_node', anonymous = False)
    ic = NCS_node()
    try:
        while(1):
            ic.ncs()
    except KeyboardInterrupt:
        print "shutting down"
    cv2.destoryAllWindows()

if __name__ == '__main__':
    main(sys.argv)
