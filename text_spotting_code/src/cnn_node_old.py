#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image

import numpy as np
from os.path import expanduser
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.msg import Rect, Rects
import caffe
import time
np.set_printoptions(threshold=np.inf)
home = expanduser("~")

class CNN_node():
    def __init__(self):
        self.image_sub = rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.img_cb)
        self.quad_sub = rospy.Subscriber("quad_proposals", Rects, self.img_crop)
        self.image_pub = rospy.Publisher('gray', Image, queue_size=10)
        self.bridge = CvBridge()
        self.cv_image = 0
        self.cv_img_crop = []
        self.start = 0
        #caffe params
        self.model = model = 'street_en_harvest'
        self.caffe_root = home +'/caffe'
        sys.path.insert(0, self.caffe_root + 'python')
        caffe.set_mode_gpu()
        self.net_full_conv = caffe.Net(home+'/models/'+self.model+'.prototxt', home+'/models/'+self.model+'.caffemodel', caffe.TEST)
        nx, ny = (3, 28)
        x = np.linspace(0, 0, nx)
        y = np.linspace(50, 255, ny)
        xv, yv = np.meshgrid(x, y)
        self.class_colors = np.zeros((29,3))
        self.class_colors[0:28]=yv
        self.class_colors[28]=[0,0,0]
        self.class_colors = self.class_colors.astype(np.uint8)
        self.colorful_class_colors = np.random.rand(29,3)*255
        self.colorful_class_colors = self.colorful_class_colors.astype(np.uint8)
        self.switch_quad = 0
        self.switch_img = 1
        self.time = 0
        self.n = 1
        
    
    def img_cb(self, data):
        #print "Image callback"
        if self.switch_img is 0:
            return
        try:
            self.start = time.time()
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #cv2.imwrite("test.jpg", self.cv_image)
            self.switch_quad = 1
            self.switch_img = 0
            
        except CvBridgeError as e:
            print(e)
      
    def img_crop(self, quads):
        if self.switch_quad is 0:
            return
        #print "Quad callback"
        for quad in quads.rects:
            #print quad
            img = self.cv_image[quad.y:quad.y+quad.h, quad.x:quad.x+quad.w]
            self.cv_img_crop.append(img)
        self.switch_quad = 0

    def cnn(self):
        if type(self.cv_image) == np.int:
            print "CNN"
            return

        if self.switch_img is not 0 or self.switch_quad is not 0:
            return
        i = 0
        for im in self.cv_img_crop:
        #im = self.cv_image
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if im is None:
                break
        #im = im/255.0
        #im = cv2.resize(im, (0,0), fx=2, fy=2)         
            im = im.reshape(im.shape[0], im.shape[1], 1)

            transformer = caffe.io.Transformer({'data': self.net_full_conv.blobs['data'].data.shape})
            transformer.set_transpose('data', (2,0,1))
        #transformer.set_raw_scale('data', 255.0)
            transformed_image = transformer.preprocess('data', im)
            transformed_image -= np.mean(transformed_image)
        # make classification map by forward and print prediction indices at each location
            self.net_full_conv.blobs['data'].data[...] = transformed_image
        #self.net_full_conv.blobs['data'].data[...] = im
        #out = self.net_full_conv.forward(data=np.asarray(transformed_image))
            out = self.net_full_conv.forward()
            if i == 0:    
                self.time += (time.time() - self.start)
                print self.n, self.time
                self.n += 1
            i += 1
            top1 = out['prob'][0].argmax()
            #if out['prob'][0][top1] >= 0.9:
            #    print 'class: ',top1
    
        self.cv_img_crop = []
        self.switch_img = 1    

def main(args):
    ic = CNN_node()
    rospy.init_node('CNN_node', anonymous = True)        
    try:
        while (1):
            ic.cnn()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main(sys.argv)
