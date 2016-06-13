#!/usr/bin/env python
import roslib; roslib.load_manifest("intrusion_detection_people")
import rospy
from rospkg import RosPack

from sensor_msgs.msg import Image

from robot_talk.msg import RobotTalk
from robot_talk.proxy import RobotTalkProxy

from soma_manager.srv import SOMA2QueryObjs, SOMA2QueryObjsRequest
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
from soma_manager.srv import SOMA2InsertObjs
from soma2_msgs.msg import SOMA2Object
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
        soma_srv_name = '/soma2/query_db'
        rospy.loginfo("Waiting for SOMA query service...")
        rospy.wait_for_service(soma_srv_name)
        rospy.loginfo("Done")        
        self.soma_srv = rospy.ServiceProxy(soma_srv_name, SOMA2QueryObjs)

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
            self.blog_collection = None # only commandline reporting

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
                print "Number of detected persons:", len(self._tracker_uuids)
                for r in self.res_roi.keys():                
                    region = self.res_roi[r]
                    for ind, i in enumerate(self._ubd_pos):
                        uuid = self._tracker_uuids[ind]
                        if region.contains_point([i.position.x, i.position.y]) and uuid not in self.res_uuids:
                            print "-> RESTRICTED region:", uuid
                            self.res_uuids.append(uuid)
                            self.talk.play_random("intrusion_detection")

                for r in self.unres_roi.keys():                
                    region = self.unres_roi[r]
                    for ind, i in enumerate(self._ubd_pos):
                        uuid = self._tracker_uuids[ind]
                        if region.contains_point([i.position.x, i.position.y]) and uuid not in self.unres_uuids:
                            print "-> UNRESTRICTED REGION:", uuid
                            self.unres_uuids.append(uuid)
                            self.talk.play_random("human_aware_nav")


    def get_rois(self):
        rois = []
        try:
            req = SOMA2QueryObjsRequest()
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
            req = SOMA2QueryObjsRequest()
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
            name = sra.kb[roi_id]["name"]
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

        
    def gen_blog_entry(self, roi_id, pos_objs, neg_objs):

        print 'Region: ' + self.get_roi_name(roi_id)

        body = '### OBJECT REPORT\n\n'
        body += '- **Region:** ' + self.get_roi_name(roi_id) + '\n\n'
        body += '- **Startime:** ' + str(self.startdate) + '\n\n'
        body += '- **Endtime:** '  + str(self.enddate)  + '\n\n'
        body += '- **Summary**: <font color="green">ALLOWED ITEMS (' + str(len(pos_objs)) + ')</font>, <font color="red">NOT-ALLOWED ITEMS (' + str(len(neg_objs)) + ')</font>\n\n'


        # # Create some blog entries
        msg_store = MessageStoreProxy(collection=self.blog_collection)
        robblog_path = roslib.packages.get_pkg_dir('soma_utils') 

        world_model = World(server_host='localhost',server_port=62345)

        print "POS_OBJS:", len(pos_objs)
        for idx, obj in enumerate(pos_objs):        
            try:
                o = world_model.get_object(obj.id)
                print idx, obj.id, obj.type
                observations = o._observations
                obs = observations[0]
            except:
                rospy.logerr("Object not in world model: %s, %s", idx, obj.id)

            # CHECK that objservation is within timeframe
            if self.start < int(obs.stamp) and int(obs.stamp) < self.end:

                rgb_mask = obs.get_message("rgb_mask")
                bridge = CvBridge()
                im = bridge.imgmsg_to_cv2(rgb_mask, desired_encoding="bgr8")
                imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                ret,thresh = cv2.threshold(imgray,127,255,0)
                contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                full_scene = obs.get_message("/head_xtion/rgb/image_rect_color")
                bridge = CvBridge()
                cv_image = bridge.imgmsg_to_cv2(full_scene, desired_encoding="bgr8")

                cv2.drawContours(cv_image,contours,-1,(0,255,0),2)
                full_scene_contour = bridge.cv2_to_imgmsg(cv_image)

                #success = cv2.imwrite(obj.id+'.jpeg',cv_image)
                img_id = msg_store.insert(full_scene_contour)
                body += '<font color="green">ALLOWED ITEM (' + str(idx+1) + '/'+ str(len(pos_objs)) + '):</font> ' + obj.type + '\n\n![My helpful screenshot](ObjectID(%s))\n\n' % img_id
                
            else:
                rospy.logerr("Ignore old observation for object: %s", obj.id)

        print "NEG_OBJS:", len(neg_objs)
        for idx, obj in enumerate(neg_objs):        
            try:
                o = world_model.get_object(obj.id)
                print idx, obj.id, obj.type
                observations = o._observations
                obs = observations[0]
            except:
                rospy.logerr("Object not in world model: %s, %s", idx, obj.id)

            # CHECK that objservation is within timeframe
            if self.start < int(obs.stamp) and int(obs.stamp) < self.end:

                rgb_mask = obs.get_message("rgb_mask")
                bridge = CvBridge()
                im = bridge.imgmsg_to_cv2(rgb_mask, desired_encoding="bgr8")
                imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                ret,thresh = cv2.threshold(imgray,127,255,0)
                contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                full_scene = obs.get_message("/head_xtion/rgb/image_rect_color")
                bridge = CvBridge()
                cv_image = bridge.imgmsg_to_cv2(full_scene, desired_encoding="bgr8")

                cv2.drawContours(cv_image,contours,-1,(0,0,255),2)
                full_scene_contour = bridge.cv2_to_imgmsg(cv_image)

                #success = cv2.imwrite(obj.id+'.jpeg',cv_image)
                img_id = msg_store.insert(full_scene_contour)
                body += '<font color="red">NOT-ALLOWED ITEM (' + str(idx+1) + '/'+ str(len(neg_objs)) + '):</font> ' + obj.type + '\n\n![My helpful screenshot](ObjectID(%s))\n\n' % img_id
                
            else:
                rospy.logerr("Ignore old observation for object: %s", obj.id)

        if len(pos_objs) > 0 or len(neg_objs) > 0:
            e = RobblogEntry(title=str(self.enddate) + " " + self.get_roi_name(roi_id), body= body )
            msg_store.insert(e)


    def gen_cmdline_report(self, roi_id, pos_objs, neg_objs):

        # GENERATE REPORT 
        print
        print 80 * "="
        print "Region:", self.get_roi_name(roi_id)
        print "Start :", self.startdate
        print "End   :", self.enddate 
        print
        print "POSITIVE objects:", len(pos_objs)
        for idx, obj in enumerate(pos_objs):
                print idx, obj.type 
        print "NEGATIVE objects:", len(neg_objs)
        for idx, obj in enumerate(neg_objs):
                print idx, obj.type
        print 80 * "="
        print

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
