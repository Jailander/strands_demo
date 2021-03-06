#!/usr/bin/env python

import sys
import rospy
from strands_executive_msgs import task_utils
from strands_executive_msgs.msg import Task
from strands_executive_msgs.srv import DemandTask, SetExecutionStatus
# import strands_executive_msgs

def get_services():
    # get services necessary to do the jon
    demand_task_srv_name = '/task_executor/demand_task'
    set_exe_stat_srv_name = '/task_executor/set_execution_status'
    rospy.loginfo("Waiting for task_executor service...")
    rospy.wait_for_service(demand_task_srv_name)
    rospy.wait_for_service(set_exe_stat_srv_name)
    rospy.loginfo("Done")
    add_tasks_srv = rospy.ServiceProxy(demand_task_srv_name, DemandTask)
    set_execution_status = rospy.ServiceProxy(set_exe_stat_srv_name, SetExecutionStatus)
    return add_tasks_srv, set_execution_status


if __name__ == '__main__':
    rospy.init_node("demand_sweep")

    if len(sys.argv)==2:
        waypoint=sys.argv[1]
        print "Using waypoint: ", waypoint
    else:
        print "Usage: demand_sweep.py waypoint"
        sys.exit(1)


    # get services to call into execution framework
    demand_task, set_execution_status = get_services()

    # Set the task executor running (if it isn't already)
    set_execution_status(True)
    print 'set execution'


    sweep_task = Task(action='do_sweep', max_duration=rospy.Duration(60), start_node_id=waypoint)
    task_utils.add_string_argument(sweep_task, 'short')
    resp = demand_task(sweep_task)
    print 'demanded task as id: %s' % resp.task_id
    rospy.loginfo('Success: %s' % resp.success)
    rospy.loginfo('Wait: %s' % resp.remaining_execution_time)

