#!/usr/bin/env python
import roslib; roslib.load_manifest("change_detection_objects")
import rospy
from rospkg import RosPack

from std_msgs.msg import Empty

from sensor_msgs.msg import Image

from robot_talk.msg import RobotTalk
from robot_talk.proxy import RobotTalkProxy

from soma_manager.srv import SOMAQueryROIs, SOMAQueryROIsRequest
from soma_io.observation import Observation
from soma_io.state import World, Object

from mongodb_store.message_store import MessageStoreProxy
from robblog.msg import RobblogEntry
import robblog.utils

from semantic_map_publisher.srv import WaypointInfoService, WaypointInfoServiceResponse,  WaypointInfoServiceRequest

from object_manager.srv import DynamicObjectsService, DynamicObjectsServiceResponse, DynamicObjectsServiceRequest

from object_manager.srv import GetDynamicObjectService, GetDynamicObjectServiceResponse, GetDynamicObjectServiceRequest

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

class ChangeDetector():

    def __init__(self, config_file=None, blog=None):    
        soma_srv_name = '/soma/query_rois'
        rospy.loginfo("Waiting for SOMA query service...")
        rospy.wait_for_service(soma_srv_name)
        rospy.loginfo("Done")        
        self.soma_srv = rospy.ServiceProxy(soma_srv_name, SOMAQueryROIs)

        if config_file:
            self._config_file = config_file
        else:
            # default file
            rp = RosPack()
            path = rp.get_path('change_detection_objects') + '/config/'
            filename = 'default.json'
            self._config_file=path+filename

        rospy.loginfo("Use KB at: %s", self._config_file)
        self._init_kb()
        self._init_rois()

        if blog:
            self.blog_collection = blog
        else:
            self.blog_collection = 'soma_blog' # only commandline reporting

        self.talk = RobotTalkProxy('robot_talk')
        
        self._setup_services()

        self._init_meta_room_count()

        
    def _init_kb(self):
        # read KB from config 
        with open(self._config_file) as config_file:
            config = json.load(config_file)
            self.kb = config

    def _init_rois(self):

        self.roi = {}
        self.roi_name = {}
        rois = self.get_rois()
        for r in rois:
            for waypoint in self.kb.keys():
                if self.kb[waypoint]["roi_id"] == r.id:
                    region = self.generate_region(r.posearray.poses)
                    self.roi[r.id] = region
                    self.roi_name[r.id] =  self.kb[waypoint]["roi_name"]
        print "ROIs", self.roi

    def _init_meta_room_count(self):

        self.count = {}
        
        resp = self.waypoint_info()

        for idx, wp in enumerate(resp.waypoint_id):
            if wp in self.kb.keys():
                self.count[wp] = resp.observation_count[idx]  - 1 # report latest change straight away 
        
        print self.count

    def has_new_observation(self, waypoint):

        resp = self.waypoint_info()
        for idx, wp in enumerate(resp.waypoint_id):
            if wp == waypoint:
                if resp.observation_count[idx] > self.count[wp]:
                    self.count[wp] = resp.observation_count[idx]
                    return True
        return False
        
    def generate_region(self, poses):
        xs = []
        ys = []
        for p in poses:
            xs.append(p.position.x)
            ys.append(p.position.y)
        region = get_polygon(xs, ys)
        return region

    def _setup_services(self):

        rospy.loginfo("Setting up change detection services")
        try:
            self.waypoint_info = rospy.ServiceProxy("/semantic_map_publisher/SemanticMapPublisher/WaypointInfoService", WaypointInfoService)
            rospy.loginfo("Wait for WaypointInfoService")
            self.waypoint_info.wait_for_service()
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s" % e)

        try:
            self.dyn_objs = rospy.ServiceProxy("/object_manager_node/ObjectManager/DynamicObjectsService", DynamicObjectsService)
            rospy.loginfo("Wait for DynamicObjectService")
            self.dyn_objs.wait_for_service()
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s" % e)

        try:
            self.get_dyn_obj = rospy.ServiceProxy("/object_manager_node/ObjectManager/GetDynamicObjectService", GetDynamicObjectService)
            rospy.loginfo("Wait for DynamicObjectService")
            self.get_dyn_obj.wait_for_service()
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s" % e)

        rospy.loginfo("Done.")
            
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

            
    def get_rois(self):
        rois = []
        try:
            req = SOMAQueryROIsRequest()
            req.query_type = 2
            rospy.loginfo("Requesting ROIs")
            res = self.soma_srv(req)
            rois = res.rois
            rospy.loginfo("Received ROIs: %s", len(res.rois))
            
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s"%e)

        return rois

    def get_roi_name(self, roi_id):
        if roi_id not in self.roi_name.keys():
            rospy.logerr("ROI is not kmown: %s", roi_id)
            return "UNKNOWN-ROI"

        name = self.roi_name[roi_id]
        return name
            
    def get_objects(self, waypoint):
        req = DynamicObjectsServiceRequest()
        req.waypoint_id = waypoint
        resp = self.dyn_objs(req)
        return resp

    def get_object(self, waypoint, obj):
        req = GetDynamicObjectServiceRequest()
        req.waypoint_id = waypoint
        req.object_id = obj
        resp = self.get_dyn_obj(req)
        return resp
        
    def gen_blog_entry(self, roi_id, obj_id, objs):

        print 'Region: ' + self.get_roi_name(roi_id)
        time = dt.fromtimestamp(int(rospy.get_time()))
        body = '### CHANGE DETECTION REPORT\n\n'
        body += '- **Region:** ' + self.get_roi_name(roi_id) + '\n\n'
        #body += '- **Object ID:** ' + str(obj_id)  + '\n\n'
        body += '- **Time:** ' + str(time)  + '\n\n'

        # Create some blog entries
        msg_store = MessageStoreProxy(collection=self.blog_collection)
        robblog_path = roslib.packages.get_pkg_dir('soma_utils') 

        for idx,obj in enumerate(objs):
            
            bridge = CvBridge()
            im = bridge.imgmsg_to_cv2(obj.image_mask, desired_encoding="bgr8")
            imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(imgray,1,1,1) #cv2.threshold(imgray,127,255,0)
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(obj.image_full, desired_encoding="bgr8")
            
            cv2.drawContours(cv_image,contours,-1,(0,0,255),2)
            full_scene_contour = bridge.cv2_to_imgmsg(cv_image)
            img_id = msg_store.insert(full_scene_contour)

            body += '<font color="red">Detected change (' + str(idx+1) + '/'+ str(len(objs)) + '):</font>\n\n![My helpful screenshot](ObjectID(%s))\n\n' % img_id
            
        e = RobblogEntry(title=str(time) + " Change Detection - " + self.roi_name[roi_id], body= body )
        msg_store.insert(e)

    
    def run(self):
        rospy.loginfo("Change detection for objects running...")
        while not rospy.is_shutdown():
            for wp in self.kb.keys():
                # if wp has no new observation ignore WP
                print "Checking for new observtions at:", wp
                if not self.has_new_observation(wp):
                    continue
                print "-> Found new observation"
                print "Checking for dynamic objects"
                objects = self.get_objects(wp)
                print "-> Found objects:", len(objects.object_id)
                objs = []
                for idx, obj in enumerate(objects.object_id):
                    # is cluster in ROI? Ignore things that are outside the region
                    # get clusterinfo/images for WP
                    centroid = objects.centroids[idx]
                    # if object not in ROI => ignore object
                    region = self.roi[self.kb[wp]['roi_id']]
                    print "Checking if object is in ROI"
                    if not region.contains_point([centroid.x, centroid.y]):
                        print "-> object not in ROI, discard object"
                        continue
                    print "-> object in ROI, get object info"
                    o = self.get_object(wp, obj)
                    #print o.orig_image
                    objs.append(o)
                    # get image mask (get it from the latched topic)
                    #img = rospy.wait_for_message('/object_manager/requested_object_mask', Image, 10)
                    # generate blog entry
                #self.gen_blog_entry(self.kb[wp]['roi_id'], obj, objs)
                    
            rospy.sleep(5.) # sleep for some time
        rospy.loginfo("Change detection for objects finished")   
        

if __name__ == '__main__':
    rospy.init_node("change_detector_objects")

    parser = argparse.ArgumentParser(prog='change_detector.py')
    parser.add_argument('-kb', metavar='<ROI-KB>')
    parser.add_argument('-blog', metavar='<blog-store>')

    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    
    change_detector  = ChangeDetector(args.kb, args.blog)
    change_detector.run()

