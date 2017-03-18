#!/usr/bin/env python
import roslib; roslib.load_manifest("intrusion_detection_people")
import rospy
from rospkg import RosPack

from sensor_msgs.msg import Image

from robot_talk.msg import RobotTalk
from robot_talk.proxy import RobotTalkProxy

from soma_manager.srv import SOMAQueryObjs, SOMAQueryObjsRequest
from soma_io.observation import Observation
from soma_io.state import World, Object

from mongodb_store.message_store import MessageStoreProxy
from robblog.msg import RobblogEntry
import robblog.utils

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


import sys
import argparse
import json
import time
from datetime import datetime as dt

import tf
import message_filters
from std_msgs.msg import *
from sensor_msgs.msg import *

from bayes_people_tracker.msg import PeopleTracker
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, JointState
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from soma_manager.srv import SOMAInsertObjs
from soma_msgs.msg import SOMAObject
from vision_people_logging.srv import CaptureUBD

import math
import itertools
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.path as mathpath


# # Implementation of Shoelace formula
# # http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# # Finding the right polygon in case the lists of xs, ys are not properly ordered
def get_polygon(xs, ys):
    if poly_area(np.array(xs), np.array(ys)) == 0.0:
        xs = [
            [xs[0]] + list(i) for i in itertools.permutations(xs[1:])
        ]
        ys = [
            [ys[0]] + list(i) for i in itertools.permutations(ys[1:])
        ]
        areas = list()
        for ind in range(len(xs)):
            areas.append(poly_area(np.array(xs[ind]), np.array(ys[ind])))
        return mathpath.Path(
            np.array(zip(xs[areas.index(max(areas))], ys[areas.index(max(areas))]))
        )
    else:
        return mathpath.Path(np.array(zip(xs, ys)))

class IntrusionDetector():

    def __init__(self, config_file=None, blog=None):    
        soma_srv_name = '/soma/query_objects'
        rospy.loginfo("Waiting for SOMA query service...")
        rospy.wait_for_service(soma_srv_name)
        rospy.loginfo("Done")        
        self.soma_srv = rospy.ServiceProxy(soma_srv_name, SOMAQueryObjs)

        if config_file:
            self._config_file = config_file
        else:
            # default file
            rp = RosPack()
            path = rp.get_path('intrusion_detection_people') + '/config/'
            filename = 'default.json'
            self._config_file=path+filename

        rospy.loginfo("Use KB at: %s", self._config_file)
        self._init_kb()
        self._init_rois()

        self.res_uuids = []
        self.unres_uuids = []
        
        if blog:
            self.blog_collection = blog
        else:
            self.blog_collection = 'soma_blog' # only commandline reporting

        self.talk = RobotTalkProxy('robot_talk')
        
        self._setup_callbacks()

        
    def _init_kb(self):
        # read KB from config 
        with open(self._config_file) as config_file:
            config = json.load(config_file)
            self.kb = config

    def _init_rois(self):

        # rois not in the config are ignored
        self.res_roi   = {}
        self.unres_roi = {}
        
        rois = self.get_rois()
        for r in rois:
            if r.id in self.kb.keys():
                if self.kb[r.id]['intrusion_detection'] == 'true':
                    region = self.generate_region(r.posearray.poses)
                    self.res_roi[r.id] = region
                elif self.kb[r.id]['intrusion_detection'] == 'false':
                    region = self.generate_region(r.posearray.poses)
                    self.unres_roi[r.id] = region
        print "res", self.res_roi
        print "unres", self.unres_roi

    def generate_region(self, poses):
        xs = []
        ys = []
        for p in poses:
            xs.append(p.position.x)
            ys.append(p.position.y)

        region = get_polygon(xs, ys)
        return region

    def _setup_callbacks(self):

        rospy.loginfo("Setting up people perception.")
        self.is_occupied = False
        self.uuids = list()
        self._ubd_pos = list()
        self._tracker_pos = list()
        self._tracker_uuids = list()
        self._tfl = tf.TransformListener()

  #      try:
  #          self.ubd_srv = rospy.ServiceProxy("/vision_logging_service/capture", CaptureUBD)
  #          self.ubd_srv.wait_for_service()
  #      except rospy.ServiceException, e:
  #          rospy.logerr("Service call failed: %s" % e)

        self.robot_pose = Pose()
        rospy.Subscriber("/robot_pose", Pose, self.robot_cb, None, 10)

        self.subs = [
            message_filters.Subscriber(
                rospy.get_param("~ubd_topic", "/upper_body_detector/bounding_box_centres"),
                PoseArray
            ),
            message_filters.Subscriber(
                rospy.get_param("~tracker_topic", "/people_tracker/positions"),
                PeopleTracker
            )
        ]
        ts = message_filters.ApproximateTimeSynchronizer(
            self.subs, queue_size=5, slop=0.15
        )
        ts.registerCallback(self.cb)

        rospy.loginfo("Done.")

    def robot_cb(self, pose):
        self.robot_pose = pose

    def cb(self, ubd_cent, pt):
        if not self.is_occupied:
            self.is_occupied = True
            self._tracker_uuids = pt.uuids
            self._ubd_pos = self.to_world_all(ubd_cent)
            self._tracker_pos = [i for i in pt.poses]
            self.analyze_detections()
            self.is_occupied = False
            
    def to_world_all(self, pose_arr):
        transformed_pose_arr = list()
        try:
            fid = pose_arr.header.frame_id
            for cpose in pose_arr.poses:
                ctime = self._tfl.getLatestCommonTime(fid, "/map")
                pose_stamped = PoseStamped(Header(1, ctime, fid), cpose)
                # Get the translation for this camera's frame to the world.
                # And apply it to all current detections.
                tpose = self._tfl.transformPose("/map", pose_stamped)
                transformed_pose_arr.append(tpose.pose)
        except tf.Exception as e:
            rospy.logwarn(e)
            # In case of a problem, just give empty world coordinates.
            return []
        return transformed_pose_arr

            
    def analyze_detections(self):
        
            if len(self._tracker_uuids) > 0:
                #print "Number of detected persons:", len(self._tracker_uuids)
                for r in self.res_roi.keys():                
                    region = self.res_roi[r]
                    for i in self._ubd_pos:
                        for ind, j in enumerate(self._tracker_pos):
                            # conditions to make sure that a person is not detected
                            # twice and can be verified by UBD logging, also is inside
                            # the surface (or target) region
                            conditions = euclidean(
                                [i.position.x, i.position.y], [j.position.x, j.position.y]
                            ) < 0.3
                            uuid = self._tracker_uuids[ind]
                            conditions = conditions and uuid not in self.res_uuids
                            conditions = conditions and region.contains_point([i.position.x, i.position.y])
                            if conditions:
                                print "-> RESTRICTED region:", uuid
                                self.res_uuids.append(uuid)
                                #print self.talk.get_random_text("intrusion_detection")
                                self.talk.play_random("intrusion_detection")
                                self.gen_blog_entry(r, uuid)

                for r in self.unres_roi.keys():                
                    region = self.unres_roi[r]
                    for i in self._ubd_pos:
                        for ind, j in enumerate(self._tracker_pos):
                            # conditions to make sure that a person is not detected
                            # twice and can be verified by UBD logging, also is inside
                            # the surface (or target) region
                            conditions = euclidean(
                                [i.position.x, i.position.y], [j.position.x, j.position.y]
                            ) < 0.3
                            uuid = self._tracker_uuids[ind]
                            conditions = conditions and uuid not in self.unres_uuids
                            conditions = conditions and region.contains_point([i.position.x, i.position.y])
                            if conditions:
                                print "-> UNRESTRICTED region:", uuid
                                self.unres_uuids.append(uuid)
                                #print self.talk.get_random_text("human_aware_nav")
                                self.talk.play_random("human_aware_nav")


    def get_rois(self):
        rois = []
        try:
            req = SOMAQueryObjsRequest()
            req.query_type = 2
            rospy.loginfo("Requesting ROIs")
            res = self.soma_srv(req)
            rois = res.rois
            rospy.loginfo("Received ROIs: %s", len(res.rois))
            
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s"%e)

        return rois

    def get_objects(self, roi_id):

        try:
            req = SOMAQueryObjsRequest()
            req.query_type = 0 
            req.useroi = True
            req.roi_id = str(roi_id)
            req.usedates = True
            req.lowerdate = int(self.start) * 1000 # MongoDB requires time in miliseconds
            req.upperdate = int(self.end) * 1000 # MongoDB requires time in miliseconds

            rospy.loginfo("Requesting objects")
            res = self.soma_srv(req)
            rospy.loginfo("Received objects: %s", len(res.objects))
            
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s"%e)

        return res

    def get_roi_name(self, roi_id):
        if roi_id not in self.kb:
            rospy.logerr("ROI is not kmown: %s", roi_id)
            return "UNKNOWN-ROI"

        name = "UNNAMED-ROI"
        if "name" in self.kb[roi_id]:
            name = self.kb[roi_id]["name"]
        return name
            
    def analyze(self, roi_id, res):

        pos_objs = []
        neg_objs = []
        
        pos_res = []
        neg_res = []

        # COUNT POS/NEG OBJECTS
        if roi_id not in self.kb:
            rospy.logerr("ROI is not kmown: %s", roi_id)
            return pos_res, neg_res
        

        if "pos_objects" in self.kb[roi_id]: 
            pos_objs = self.kb[roi_id]["pos_objects"]

        if "neg_objects" in self.kb[roi_id]: 
            neg_objs = self.kb[roi_id]["neg_objects"]

        for idx, o in enumerate(res.objects):
            obj = res.objects[idx]
            if obj.type == "person":
                continue
            
            if obj.type in neg_objs or (neg_objs == [] and obj.type not in pos_objs):
                neg_res.append(obj)

            elif obj.type in pos_objs or (pos_objs == [] and obj.type not in neg_objs):
                pos_res.append(obj)
            

        return pos_res, neg_res

        
    def gen_blog_entry(self, roi_id, uuid):

        print 'Region: ' + self.get_roi_name(roi_id)
        time = dt.fromtimestamp(int(rospy.get_time()))
        body = '### INTRUSION DETECTION REPORT\n\n'
        body += '- **Region:** ' + self.get_roi_name(roi_id) + '\n\n'
        body += '- **Person UUID:** ' + str(uuid)  + '\n\n'
        body += '- **Time:** ' + str(time)  + '\n\n'
        #body += '- **Summary**: <font color="green">ALLOWED ITEMS (' + str(len(pos_objs)) + ')</font>, <font color="red">NOT-ALLOWED ITEMS (' + str(len(neg_objs)) + ')</font>\n\n'


        # # Create some blog entries
        msg_store = MessageStoreProxy(collection=self.blog_collection)
        robblog_path = roslib.packages.get_pkg_dir('soma_utils') 

        img = rospy.wait_for_message('/upper_body_detector/image', Image, 5)

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        ros_img = bridge.cv2_to_imgmsg(cv_image)
        img_id = msg_store.insert(ros_img)
        body += '<font color="red">Detected person:</font>\n\n![My helpful screenshot](ObjectID(%s))\n\n' % img_id

        e = RobblogEntry(title=str(time) + " Intrusion Detection - " + self.get_roi_name(roi_id), body= body )
        msg_store.insert(e)

    
    def run(self):
        rospy.loginfo("Intrusion detection for people running...")
        rospy.spin()
        rospy.loginfo("Intrusion detection for people has finished")   
    


        

if __name__ == '__main__':
    rospy.init_node("intrusion_detector_people")

    parser = argparse.ArgumentParser(prog='intrusion_detector.py')
    parser.add_argument('-kb', metavar='<ROI-KB>')
    parser.add_argument('-blog', metavar='<blog-store>')

    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    
    intrusion_detector  = IntrusionDetector(args.kb, args.blog)
    intrusion_detector.run()

