#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:32:28 2020

@author: patryk
"""
#from pymitsubishi_controller.controller.joint import MitsubishiPythonJointController
import threading
from mujoco_py import MjSim, MjViewer, load_model_from_path, const
import numpy as np
from collections import deque
import time
import random
import os
import gc
import math
from timeit import default_timer as timer 
from numba import jit, cuda 
import sys
import readchar
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
import glfw
import pyzed.sl as sl
import cv2
import tensorflow as tf
from BackPropagationNN import NeuralNetwork as NN
from GeneticAlgorithm import Genetic as genetic

nn = NN(20, 14, 2)
nn.ReadWeights()
param_nn = []
#ladowanie modelu sprawdzanie testy
model = load_model_from_path("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/assets/mitsubishi_rv_12sdl_6DOF_new_model_gripper93_mocap.xml")
sim = MjSim(model)
viewer = 0

model1 = load_model_from_path("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/assets/mitsubishi_rv_12sdl_6DOF_new_model_gripper93_mocap.xml")
sim1 = MjSim(model1)


viewer1 = MjViewer(sim1)
renders = 0
while renders<2:
    viewer1.render()
    time.sleep(0.1)
    renders = renders + 1
window1 = glfw.get_current_context()

glfw.destroy_window(window1)
  
        

joints = [n for n in sim.model.joint_names if n.startswith('robot0')]
print(joints)

gripper_pos = np.array(sim.data.get_body_xpos("robot0:gripper"))
sim.data.set_mocap_pos('robot0:mocap', gripper_pos)
print(gripper_pos)

data_x_y_z = sim.data.get_body_xpos("robot0:gripper")
data_x_y_z1 = sim.data.get_body_xpos("robot0:wrist_flange")
        
print(data_x_y_z)
print(data_x_y_z1)
    
list_of_action_six_axis = []

#variable uses to save data
x_chart = []
y_chart = []
z_chart = []
j1_rad = []
j2_rad = []
j3_rad = []
j4_rad = []
j5_rad = []
j6_rad = []


err11 = 1                       #trzeba dac osobne neurony
err22 = 1
err33 = 1
err44 = 1
err55 = 1
err66 = 1   
big_step_main = 10.0#7.8001159534344655
little_step_main = 2.0#0.5298340969323152
'''
enemy1_pos = np.array([900, 75, 650])
enemy2_pos = np.array([850, -75, 600])
hand_pos = np.array([700, 200, 480])
object_to_move_pos = np.array([900, -200, 580])
'''
object_to_move_pos = np.array([608, -60, 627]) 
hand_pos = np.array([997, 198, 417])
enemy1_pos = np.array([823, -110,  700])
enemy2_pos = np.array([784, 67, 600])

TEST = True  

start_motion = 0
teaching_first_motion = 0
teaching_second_motion = 0
teaching_third_motion = 0
steps1 = 0
steps2 = 0
steps3 = 0
steps4 = 0
steps5 = 0
steps6 = 0
steps7 = 0
RESERVE_MM = 100
start_pos_of_motion = []
start_pos_of_motion.insert(0, 0)
start_pos_of_motion.insert(1, 0)
start_pos_of_motion.insert(2, 0)
start_pos_of_motion.insert(3, 0)
start_pos_of_motion.insert(4, 0)
start_pos_of_motion.insert(5, 0)
start_pos_first_motion = []
start_pos_second_motion = []
start_pos_third_motion = []
start_pos_fourth_motion = []
start_pos_fifth_motion = []
start_pos_sixth_motion = []
start_pos_seventh_motion = []
action1 = []
action2 = []
action3 = []
action4 = []
action5 = []
action6 = []
action7 = []



#Wypelnianie tablicy z ruchami wylaczanie wlaczanie osi i zmiana rozdzielczosci ruchu
ACTION_SIZE = 64     
@jit
def ActionInit(multiply_degree, active1, active2, active3, active4, active5, active6):
    list_of_action_six_axis.clear()
    DEG_RESOLUTION1 = 0.00174 * multiply_degree * active1 #0.0174 = 1 degree
    DEG_RESOLUTION2 = 0.00174 * multiply_degree * active2 #0.0174 = 1 degree
    DEG_RESOLUTION3 = 0.00174 * multiply_degree * active3 #0.0174 = 1 degree
    DEG_RESOLUTION4 = 0.00174 * multiply_degree * active4 #0.0174 = 1 degree
    DEG_RESOLUTION5 = 0.00174 * multiply_degree * active5 #0.0174 = 1 degree
    DEG_RESOLUTION6 = 0.00174 * multiply_degree * active6 #0.0174 = 1 degree
    for joint1_direction in range(-1, 2, 2):
        for joint2_direction in range(-1, 2, 2):
            for joint3_direction in range(-1, 2, 2):
                for joint4_direction in range(-1, 2, 2):
                    for joint5_direction in range(-1, 2, 2):
                         for joint6_direction in range(-1, 2, 2):
                             list_of_action_six_axis.append([joint1_direction*DEG_RESOLUTION1, joint2_direction*DEG_RESOLUTION2, joint3_direction*DEG_RESOLUTION3, 
                                                        joint4_direction*DEG_RESOLUTION4, joint5_direction*DEG_RESOLUTION5, joint6_direction*DEG_RESOLUTION6])#joint4_direction*DEG_RESOLUTION


                


@jit
def DistPoinToPoint(first_point, second_point):
    distance = math.sqrt((first_point[0]-(second_point[0]*1000))**2 + (first_point[1]-(second_point[1]*1000))**2 + (first_point[2] -(second_point[2]*1000))**2)
    
    return distance
 
@jit  
def DistLineToLine(line_start_enemy, line_end_enemy, line_start_robot_point, line_end_robot_point, width_enemy):
    last_closest_dist = 3000
    line_dot_robot = []
    line_dot_robot.insert(0, 0)
    line_dot_robot.insert(1, 0)
    line_dot_robot.insert(2, 0)
    line_dot_enemy = []
    line_dot_enemy.insert(0, 0)
    line_dot_enemy.insert(1, 0)
    line_dot_enemy.insert(2, 0)
    t = 0
    for e in range(11):
        t1 = 0
        t = t+(e/10)
        line_dot_robot[0] = line_start_robot_point[0] + (line_end_robot_point[0] - line_start_robot_point[0])*t
        line_dot_robot[1] = line_start_robot_point[1] + (line_end_robot_point[1] - line_start_robot_point[1])*t
        line_dot_robot[2] = line_start_robot_point[2] + (line_end_robot_point[2] - line_start_robot_point[2])*t
        for z in range(11):
            t=0
            t1 =t1+(z/10)
            line_dot_enemy[0] = line_start_enemy[0] + (line_end_enemy[0] - line_start_enemy[0])*t1
            line_dot_enemy[1] = line_start_enemy[1] + (line_end_enemy[1] - line_start_enemy[1])*t1
            line_dot_enemy[2] = (line_start_enemy[2]-width_enemy) + (line_end_enemy[2] - (line_start_enemy[2]-width_enemy))*t1
            distance = math.sqrt((line_dot_robot[0]-line_dot_enemy[0])**2 + (line_dot_robot[1]-line_dot_enemy[1])**2 + (line_dot_robot[2] -line_dot_enemy[2])**2)
            if (distance < last_closest_dist):
                last_closest_dist = distance
            
    return last_closest_dist

@jit
def PrepeareAllLineEnemyAndCheckDistance(enemy1_start_line, enemy1_end_line, enemy2_start_line, enemy2_end_line, enemy_startboxfirst_pos, enemy_endboxfirst_pos, enemy_startboxsecond_pos, enemy_endboxsecond_pos,
                                         pos1_robot, pos2_robot, pos3_robot, pos4_robot, pos5_robot, pos6_robot, width_enemy, get_object, put_object):
    return_val = True
    
    dist = []
    dist.insert(0, 0)
    dist.insert(1, 0)
    dist.insert(2, 0)
    dist.insert(3, 0)
    dist.insert(4, 0)
    dist.insert(5, 0)
    dist.insert(6, 0)
    dist.insert(7, 0)
    
    dist.insert(8, 0)
    dist.insert(9, 0)
    dist.insert(10, 0)
    dist.insert(11, 0)
    dist.insert(12, 0)
    dist.insert(13, 0)
    dist.insert(14, 0)
    dist.insert(15, 0)
    
    dist.insert(16, 0)
    dist.insert(17, 0)
    dist.insert(18, 0)
    dist.insert(19, 0)
    
    
    dist[0] = DistLineToLine(enemy1_start_line*1000, enemy1_end_line*1000, pos1_robot*1000, pos2_robot*1000, width_enemy)
    #print("Dist 0: " + str(dist[0]))
    dist[1] = DistLineToLine(enemy1_start_line*1000, enemy1_end_line*1000, pos2_robot*1000, pos3_robot*1000, width_enemy)
    #print("Dist 1: " + str(dist[1]))
    dist[2] = DistLineToLine(enemy1_start_line*1000, enemy1_end_line*1000, pos3_robot*1000, pos4_robot*1000, width_enemy)
    #print("Dist 2: " + str(dist[2]))
    dist[3] = DistLineToLine(enemy2_start_line*1000, enemy2_end_line*1000, pos1_robot*1000, pos2_robot*1000, width_enemy)
    #print("Dist 3: " + str(dist[3]))
    dist[4] = DistLineToLine(enemy2_start_line*1000, enemy2_end_line*1000, pos2_robot*1000, pos3_robot*1000, width_enemy)
    #print("Dist 4: " + str(dist[4]))
    dist[5] = DistLineToLine(enemy2_start_line*1000, enemy2_end_line*1000, pos3_robot*1000, pos4_robot*1000, width_enemy)
    #print("Dist 5: " + str(dist[5]))
    dist[6] = DistLineToLine(enemy1_start_line*1000, enemy1_end_line*1000, pos5_robot*1000, pos6_robot*1000, width_enemy)
    #print("Dist 6: " + str(dist[6]))
    dist[7] = DistLineToLine(enemy2_start_line*1000, enemy2_end_line*1000, pos5_robot*1000, pos6_robot*1000, width_enemy)
    #print("Dist 7: " + str(dist[7]))
    '''
    dist[8] = DistLineToLine(enemy_startboxfirst_pos*1000, enemy_endboxfirst_pos*1000, pos1_robot*1000, pos2_robot*1000, width_enemy)
    #print("Dist 8: " + str(dist[8]))
    dist[9] = DistLineToLine(enemy_startboxfirst_pos*1000, enemy_endboxfirst_pos*1000, pos2_robot*1000, pos3_robot*1000, width_enemy)
    #print("Dist 9: " + str(dist[9]))
    dist[10] = DistLineToLine(enemy_startboxfirst_pos*1000, enemy_endboxfirst_pos*1000, pos3_robot*1000, pos4_robot*1000, width_enemy)
    #print("Dist 10: " + str(dist[10]))
    dist[11] = DistLineToLine(enemy_startboxsecond_pos*1000, enemy_endboxsecond_pos*1000, pos1_robot*1000, pos2_robot*1000, width_enemy)
    #print("Dist 11: " + str(dist[11]))
    dist[12] = DistLineToLine(enemy_startboxsecond_pos*1000, enemy_endboxsecond_pos*1000, pos2_robot*1000, pos3_robot*1000, width_enemy)
    #print("Dist 12: " + str(dist[12]))
    dist[13] = DistLineToLine(enemy_startboxsecond_pos*1000, enemy_endboxsecond_pos*1000, pos3_robot*1000, pos4_robot*1000, width_enemy)
    #print("Dist 13: " + str(dist[13]))
    dist[14] = DistLineToLine(enemy_startboxfirst_pos*1000, enemy_endboxfirst_pos*1000, pos5_robot*1000, pos6_robot*1000, width_enemy)
    #print("Dist 14: " + str(dist[14]))
    dist[15] = DistLineToLine(enemy_startboxsecond_pos*1000, enemy_endboxsecond_pos*1000, pos5_robot*1000, pos6_robot*1000, width_enemy)
    #print("Dist 15: " + str(dist[15]))
    '''
    perpendicular_pos = 0
    
    if get_object == True:
        #dist[16] = DistPoinToPoint(enemy_startboxfirst_pos, pos5_robot)
        #dist[17] = DistPoinToPoint(enemy_startboxfirst_pos, pos6_robot)
        perpendicular_pos = abs(pos5_robot[2] - pos6_robot[2])
        
    if put_object == True:
        #dist[18] = DistPoinToPoint(enemy_startboxsecond_pos, pos5_robot)
        #dist[19] = DistPoinToPoint(enemy_startboxsecond_pos, pos6_robot)
        perpendicular_pos = abs(pos5_robot[2] - pos6_robot[2])*1000
    
    enemy_dist = 0
    
    for w in range(8):
        enemy_dist = enemy_dist + dist[w] 
    enemy_dist = enemy_dist/8
    
    if dist[0] < 90:
        return_val = False   
    elif dist[1] < 150:
        return_val = False
    elif dist[2] < 195:
        return_val = False
    elif dist[3] < 90:
        return_val = False
    elif dist[4] < 175:
        return_val = False
    elif dist[5] < 195:
        return_val = False
    elif dist[6] < 75:
        return_val = False
    elif dist[7] < 75:
        return_val = False
    '''
    elif dist[8] < 60:
        return_val = False
    elif dist[9] < 150:
        return_val = False
    elif dist[10] < 150:
        return_val = False
    elif dist[11] < 60:
        return_val = False
    elif dist[12] < 150:
        return_val = False
    elif dist[13] < 150:
        return_val = False
    elif dist[14] < 65:
        return_val = False
    elif dist[15] < 65:
        return_val = False
    ''' 
    return return_val, enemy_dist, perpendicular_pos 
                             
def setPosRender(data, move_object_by_gripper):
        j = 1
        joint1_range = 1
        joint2_range = 1
        joint3_range = 1
        joint4_range = 1
        joint5_range = 1
        joint6_range = 1
        move_1_joint = data[0]+data[6]
        move_2_joint = data[1]+data[7]
        move_3_joint = data[2]+data[8]
        move_4_joint = data[3]+data[9]
        move_5_joint = data[4]+data[10]
        move_6_joint = data[5]+data[11]
        
        if move_1_joint > 0.7:
            move_1_joint = 0.7
            joint1_range = 0
        if move_1_joint < -0.7:
            move_1_joint = -0.7
            joint1_range = 0
            
        if move_2_joint > 1.0:
            move_2_joint = 1.0
            joint2_range = 0
        if move_2_joint < -0.4:
            move_2_joint = -0.4
            joint2_range = 0
            
        if move_3_joint > 1:
            move_3_joint = 1
            joint3_range = 0
        if move_3_joint < -1:
            move_3_joint = -1
            joint3_range = 0
            
        if move_4_joint > 2.79:
            move_4_joint = 2.79
            joint4_range = 0
        if move_4_joint < -2.79:
            move_4_joint = -2.79
            joint4_range = 0
            
        if move_5_joint > 0.3:
            move_5_joint = 0.3
            joint5_range = 0
        if move_5_joint < -3.14:
            move_5_joint = -3.14
            joint5_range = 0
            
        if move_6_joint > 6.28:
            move_6_joint = 6.28
            joint6_range = 0
        if move_6_joint < -6.28:
            move_6_joint = -6.28
            joint6_range = 0
        
        for i in joints:
            if(j==1):
                sim.data.set_joint_qpos(i, move_1_joint)
            if(j==2):
                sim.data.set_joint_qpos(i, move_2_joint)
            if(j==3):
                sim.data.set_joint_qpos(i, move_3_joint)
            if(j==4):
                sim.data.set_joint_qpos(i, move_4_joint)
            if(j==5):
                sim.data.set_joint_qpos(i, move_5_joint)
            if(j==6):
                sim.data.set_joint_qpos(i, move_6_joint)
            j = j + 1
        
        #if move_object_by_gripper == True:
            #data_x_y_z = sim.data.get_body_xpos("robot0:gripper")
            #sim.data.set_joint_qpos("object:x", data_x_y_z[0])
            #sim.data.set_joint_qpos("object:y", data_x_y_z[1])
            #sim.data.set_joint_qpos("object:z", data_x_y_z[2])
        #else:
            #sim.data.set_joint_qpos("object:x", 0.823)
            #sim.data.set_joint_qpos("object:y", -0.195)
            #sim.data.set_joint_qpos("object:z", 0.323)
            
        viewer.render()

        sim.step()
        data_x_y_z = sim.data.get_body_xpos("robot0:gripper")
        data_x_y_z1 = sim.data.get_body_xpos("robot0:wrist_flange")
        data_x_y_z2 = sim.data.get_body_xpos("robot0:elbow")
        data_x_y_z3 = sim.data.get_body_xpos("robot0:shoulder")
        data_x_y_z4 = sim.data.get_body_xpos("robot0:gripper1")
        data_x_y_z5 = sim.data.get_body_xpos("robot0:gripper2")
        
        
        return data_x_y_z, data_x_y_z1, data_x_y_z2, data_x_y_z3, data_x_y_z4, data_x_y_z5, joint1_range, joint2_range, joint3_range, joint4_range, joint5_range, joint6_range
    
def setPos(data, move_object_by_gripper):
        j = 1
        joint1_range = 1
        joint2_range = 1
        joint3_range = 1
        joint4_range = 1
        joint5_range = 1
        joint6_range = 1
        move_1_joint = data[0]+data[6]
        move_2_joint = data[1]+data[7]
        move_3_joint = data[2]+data[8]
        move_4_joint = data[3]+data[9]
        move_5_joint = data[4]+data[10]
        move_6_joint = data[5]+data[11]
        
        if move_1_joint > 0.7:
            move_1_joint = 0.7
            joint1_range = 0
        if move_1_joint < -0.7:
            move_1_joint = -0.7
            joint1_range = 0
            
        if move_2_joint > 1.0:
            move_2_joint = 1.0
            joint2_range = 0
        if move_2_joint < -0.4:
            move_2_joint = -0.4
            joint2_range = 0
            
        if move_3_joint > 1:
            move_3_joint = 1
            joint3_range = 0
        if move_3_joint < -1:
            move_3_joint = -1
            joint3_range = 0
            
        if move_4_joint > 2.79:
            move_4_joint = 2.79
            joint4_range = 0
        if move_4_joint < -2.79:
            move_4_joint = -2.79
            joint4_range = 0
            
        if move_5_joint > 0.3:
            move_5_joint = 0.3
            joint5_range = 0
        if move_5_joint < -3.14:
            move_5_joint = -3.14
            joint5_range = 0
            
        if move_6_joint > 6.28:
            move_6_joint = 6.28
            joint6_range = 0
        if move_6_joint < -6.28:
            move_6_joint = -6.28
            joint6_range = 0
        
        for i in joints:
            if(j==1):
                sim.data.set_joint_qpos(i, move_1_joint)
            if(j==2):
                sim.data.set_joint_qpos(i, move_2_joint)
            if(j==3):
                sim.data.set_joint_qpos(i, move_3_joint)
            if(j==4):
                sim.data.set_joint_qpos(i, move_4_joint)
            if(j==5):
                sim.data.set_joint_qpos(i, move_5_joint)
            if(j==6):
                sim.data.set_joint_qpos(i, move_6_joint)
            j = j + 1
        '''
        if move_object_by_gripper == True:
            data_x_y_z = sim.data.get_body_xpos("robot0:gripper")
            sim.data.set_joint_qpos("object:x", data_x_y_z[0])
            sim.data.set_joint_qpos("object:y", data_x_y_z[1])
            sim.data.set_joint_qpos("object:z", data_x_y_z[2])
        else:
            sim.data.set_joint_qpos("object:x", 0.823)
            sim.data.set_joint_qpos("object:y", -0.195)
            sim.data.set_joint_qpos("object:z", 0.323)
        '''
        #viewer.render()
        sim.step()
        data_x_y_z = sim.data.get_body_xpos("robot0:gripper")
        data_x_y_z1 = sim.data.get_body_xpos("robot0:wrist_flange")
        data_x_y_z2 = sim.data.get_body_xpos("robot0:elbow")
        data_x_y_z3 = sim.data.get_body_xpos("robot0:shoulder")
        data_x_y_z4 = sim.data.get_body_xpos("robot0:gripper1")
        data_x_y_z5 = sim.data.get_body_xpos("robot0:gripper2")
        
        
        return data_x_y_z, data_x_y_z1, data_x_y_z2, data_x_y_z3, data_x_y_z4, data_x_y_z5, joint1_range, joint2_range, joint3_range, joint4_range, joint5_range, joint6_range
 
#@jit    
def Workspace(self, work_x, work_y, work_z):
        status = True
        if(work_x*1000 < 400):
            status = False
        if(work_x*1000 > 1200):
            status = False
        if(work_y*1000 <-400):
            status = False
        if(work_y*1000 > 400):
            status = False
        if(work_z*1000 > 1000):
            status = False
        if(work_z*1000 < 200):
            status = False
            
        if ((work_x*1000 > 780-self.RESERVE_MM) and (work_x*1000 < 820+self.RESERVE_MM) and (work_y*1000 > -70-self.RESERVE_MM) and (work_y*1000<-30+self.RESERVE_MM) and (work_z*1000 < 700+self.RESERVE_MM)):
            status = False
            
        if ((work_x*1000 > 630-self.RESERVE_MM) and (work_x*1000 < 670+self.RESERVE_MM) and (work_y*1000 > 130-self.RESERVE_MM) and (work_y*1000< 170+self.RESERVE_MM) and (work_z*1000 < 600+self.RESERVE_MM)):
            status = False
        
        return status

def AgentSearching(param1, param2, start_motion, end_position, end_pos_accuracy_mm, devided_enemy, j_start_pos, move_object_by_gripper, get_object, put_object, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos):
    distance_last = 10000
    action_array = []
    for a in range(2000):
        action_array.append(int(0))
    next_j_pos = []
    next_j_pos.insert(0, 0)
    next_j_pos.insert(1, 0)
    next_j_pos.insert(2, 0)
    next_j_pos.insert(3, 0)
    next_j_pos.insert(4, 0)
    next_j_pos.insert(5, 0)
    
    next_j = []
    next_j.insert(0, 0)
    next_j.insert(1, 0)
    next_j.insert(2, 0)
    next_j.insert(3, 0)
    next_j.insert(4, 0)
    next_j.insert(5, 0)
    back_j = []
    inkr = -1
    back_j.insert(0,j_start_pos[0])
    back_j.insert(1,j_start_pos[1]) 
    back_j.insert(2,j_start_pos[2]) 
    back_j.insert(3,j_start_pos[3]) 
    back_j.insert(4,j_start_pos[4]) 
    back_j.insert(5,j_start_pos[5])
    for j in range(500):
        inkr = inkr + 1
        distance_last = 10000
        fail_move = 0
        #print("STEP: " + str(j))
        #print("Action: " + str(action_array[j]))
        for i in range(ACTION_SIZE): 
            #start = timer()             
            data = list_of_action_six_axis[i]
            send = []
            send.insert(0, back_j[0])
            send.insert(1, back_j[1])
            send.insert(2, back_j[2])
            send.insert(3, back_j[3])
            send.insert(4, back_j[4])
            send.insert(5, back_j[5])
            send.insert(6, data[0])
            send.insert(7, data[1])
            send.insert(8, data[2])
            send.insert(9, data[3])
            send.insert(10, data[4])
            send.insert(11, data[5])
            current_position, crash1, crash2, crash3, crash4, crash5, joint1_range, joint2_range, joint3_range, joint4_range, joint5_range, joint6_range  = setPos(send, move_object_by_gripper)
            if (joint1_range == 0):
                data[0] = 0
                
            if (joint2_range == 0):
                data[1] = 0
                
            if (joint3_range == 0):
                data[2] = 0
                
            if (joint4_range == 0):
                data[3] = 0
                
            if (joint5_range == 0):
                data[4] = 0
                
            if (joint6_range == 0):
                data[5] = 0
                
            distance = math.sqrt((end_position[0]-(current_position[0]*1000))**2 + (end_position[1]-(current_position[1]*1000))**2 + (end_position[2] -(current_position[2]*1000))**2)
            #print(str(distance))
            x = abs((end_position[0] - current_position[0]*1000))
            y = abs((end_position[1] - current_position[1]*1000))
            z = abs((end_position[2] - current_position[2]*1000))
            dist1 = abs(x-y)
            dist2 = abs(x-z)
            dist3 = abs(y-z)
            dist = dist1 + dist2 + dist3
            enemy1_start_pos = np.array([enemy1_pos[0]/1000, enemy1_pos[1]/1000, enemy1_pos[2]/1000])
            enemy1_end_pos = np.array([enemy1_pos[0]/1000, enemy1_pos[1]/1000, 0])
            enemy2_start_pos = np.array([enemy2_pos[0]/1000, enemy2_pos[1]/1000, enemy2_pos[2]/1000])
            enemy2_end_pos = np.array([enemy2_pos[0]/1000, enemy2_pos[1]/1000, 0])
            
            enemy_startboxfirst_pos = np.array([object_to_move_pos[0]/1000, object_to_move_pos[1]/1000, (object_to_move_pos[2]/1000)-0.03])
            enemy_endboxfirst_pos = np.array([object_to_move_pos[0]/1000, object_to_move_pos[1]/1000, 0])
            enemy_startboxsecond_pos = np.array([hand_pos[0]/1000, hand_pos[1]/1000, (hand_pos[2]/1000)-0.03])
            enemy_endboxsecond_pos = np.array([hand_pos[0]/1000, hand_pos[1]/1000, 0])
            result, enemy_dist, perpendicular_pos = PrepeareAllLineEnemyAndCheckDistance(enemy1_start_pos, enemy1_end_pos, enemy2_start_pos, enemy2_end_pos, enemy_startboxfirst_pos, enemy_endboxfirst_pos, enemy_startboxsecond_pos,
                                                          enemy_endboxsecond_pos, current_position, crash1, crash2, crash3, crash4, crash5, 35, get_object, put_object)

            if result == False:
                fail_move = fail_move+1
                distance = 10000
            if fail_move >= 62:
                return action_array, 1000, next_j_pos, False
            #param1, param2 =   
            reward = distance - (enemy_dist/((param1)+(j*param2))) + perpendicular_pos
            #print("time:", timer()-start)
            if (reward < distance_last):
                distance_last = reward
                action_array[j] = i
                next_j_pos[0] = back_j[0] + data[0]
                next_j_pos[1] = back_j[1] + data[1]
                next_j_pos[2] = back_j[2] + data[2]
                next_j_pos[3] = back_j[3] + data[3]
                next_j_pos[4] = back_j[4] + data[4]
                next_j_pos[5] = back_j[5] + data[5]
            if distance <= end_pos_accuracy_mm:
                return action_array, j+1, next_j_pos, True                
            if j == 499:
                print("fail" + str(start_motion))
                return action_array, 1000, next_j_pos, False
            send1 = []
            send1.insert(0, back_j[0])
            send1.insert(1, back_j[1])
            send1.insert(2, back_j[2])
            send1.insert(3, back_j[3])
            send1.insert(4, back_j[4])
            send1.insert(5, back_j[5])
            send1.insert(6, next_j[0])
            send1.insert(7, next_j[1])
            send1.insert(8, next_j[2])
            send1.insert(9, next_j[3])
            send1.insert(10, next_j[4])
            send1.insert(11, next_j[5])
            setPos(send1, move_object_by_gripper)
            #time.sleep(0.5)
        back_j.insert(0,next_j_pos[0])
        back_j.insert(1,next_j_pos[1]) 
        back_j.insert(2,next_j_pos[2]) 
        back_j.insert(3,next_j_pos[3]) 
        back_j.insert(4,next_j_pos[4]) 
        back_j.insert(5,next_j_pos[5])
    print("Fail")
        

def AgentTeaching(start_motion, end_position, end_pos_accuracy_mm, devided_enemy, j_start_pos, move_object_by_gripper, get_object, put_object, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos):
    global param_nn, err11, err22, err33, err44, err55, err66
    distance_last = 1000
    action_array = []
    for a in range(1000):
        action_array.append(0)
    next_j_pos = []
    next_j_pos.insert(0, 0)
    next_j_pos.insert(1, 0)
    next_j_pos.insert(2, 0)
    next_j_pos.insert(3, 0)
    next_j_pos.insert(4, 0)
    next_j_pos.insert(5, 0)
    
    next_j = []
    next_j.insert(0, 0)
    next_j.insert(1, 0)
    next_j.insert(2, 0)
    next_j.insert(3, 0)
    next_j.insert(4, 0)
    next_j.insert(5, 0)
    back_j = []
    inkr = -1
    back_j.insert(0,j_start_pos[0])
    back_j.insert(1,j_start_pos[1]) 
    back_j.insert(2,j_start_pos[2]) 
    back_j.insert(3,j_start_pos[3]) 
    back_j.insert(4,j_start_pos[4]) 
    back_j.insert(5,j_start_pos[5])
    
    reward_function_parameters = []
    pat = []
    param_output = []
    for j in range(500):
        inkr = inkr + 1
        distance_last = 100000
        for i in range(ACTION_SIZE): 
            #start = timer()             
            data = list_of_action_six_axis[i]
            send = []
            send.insert(0, back_j[0])
            send.insert(1, back_j[1])
            send.insert(2, back_j[2])
            send.insert(3, back_j[3])
            send.insert(4, back_j[4])
            send.insert(5, back_j[5])
            send.insert(6, data[0])
            send.insert(7, data[1])
            send.insert(8, data[2])
            send.insert(9, data[3])
            send.insert(10, data[4])
            send.insert(11, data[5])
            current_position, crash1, crash2, crash3, crash4, crash5, joint1_range, joint2_range, joint3_range, joint4_range, joint5_range, joint6_range  = setPos(send, move_object_by_gripper)
            if (joint1_range == 0):
                data[0] = 0
                
            if (joint2_range == 0):
                data[1] = 0
                
            if (joint3_range == 0):
                data[2] = 0
                
            if (joint4_range == 0):
                data[3] = 0
                
            if (joint5_range == 0):
                data[4] = 0
                
            if (joint6_range == 0):
                data[5] = 0
                
            distance = math.sqrt((end_position[0]-(current_position[0]*1000))**2 + (end_position[1]-(current_position[1]*1000))**2 + (end_position[2] -(current_position[2]*1000))**2)
            #print(str(distance))
            x = abs((end_position[0] - current_position[0]*1000))
            y = abs((end_position[1] - current_position[1]*1000))
            z = abs((end_position[2] - current_position[2]*1000))
            '''
            dist1 = abs(x-y)
            dist2 = abs(x-z)
            dist3 = abs(y-z)
            dist = dist1 + dist2 + dist3
            '''
            enemy1_start_pos = enemy1_pos/1000
            enemy1_end_pos = enemy1_pos/1000
            enemy1_end_pos[2] = 0
            enemy2_start_pos = enemy2_pos/1000
            enemy2_end_pos = enemy2_pos/1000
            enemy2_end_pos[2] = 0
            
            enemy_startboxfirst_pos = object_to_move_pos/1000
            enemy_endboxfirst_pos = object_to_move_pos/1000
            enemy_endboxfirst_pos[2] = 0
            enemy_startboxsecond_pos = hand_pos/1000
            enemy_endboxsecond_pos = hand_pos/1000
            enemy_endboxsecond_pos[2] = 0
            result, enemy_dist, perpendicular_pos = PrepeareAllLineEnemyAndCheckDistance(enemy1_start_pos, enemy1_end_pos, enemy2_start_pos, enemy2_end_pos, enemy_startboxfirst_pos, enemy_endboxfirst_pos, enemy_startboxsecond_pos,
                                                          enemy_endboxsecond_pos, current_position, crash1, crash2, crash3, crash4, crash5, 35, get_object, put_object)

            if result == False:
                distance = 100000
            if j == 0 and i == 0 and start_motion == 0:
                pat = [
                    [[err11, err22, err33, err44, err55, err66, big_step_main/1000, little_step_main/1000, enemy_startboxfirst_pos[0], enemy_startboxfirst_pos[1], enemy_startboxfirst_pos[2],
                      enemy_startboxsecond_pos[0], enemy_startboxsecond_pos[1], enemy_startboxsecond_pos[2], enemy1_start_pos[0], enemy1_start_pos[1], enemy1_start_pos[2],
                      enemy2_start_pos[0], enemy2_start_pos[1], enemy2_start_pos[2]]]
                    ]
                #print("Input NN: " + str(pat))
                param_output = nn.test(pat)
                param_nn = param_output
                print(str(param_output))
            if j == 0 and i == 0 and start_motion == 2:
                pat = [
                    [[err11, err22, err33, 0, err55, 0, big_step_main/1000, little_step_main/1000, enemy_startboxfirst_pos[0], enemy_startboxfirst_pos[1], enemy_startboxfirst_pos[2],
                      enemy_startboxsecond_pos[0], enemy_startboxsecond_pos[1], enemy_startboxsecond_pos[2], enemy1_start_pos[0], enemy1_start_pos[1], enemy1_start_pos[2],
                      enemy2_start_pos[0], enemy2_start_pos[1], enemy2_start_pos[2]]]
                    ]
                #print("Input NN: " + str(pat))
                param_output = nn.test(pat)
                param_nn = param_output
                print(str(param_output))
            reward = distance - (enemy_dist/((param_nn[0])+(j*param_nn[1]))) + perpendicular_pos
            #reward = distance - (enemy_dist/((0.7151636601088712)+(j*0.08861971669337249))) + perpendicular_pos
            #print("time:", timer()-start)
            if (reward < distance_last):
                distance_last = reward
                action_array[j] = i
                next_j_pos[0] = back_j[0] + data[0]
                next_j_pos[1] = back_j[1] + data[1]
                next_j_pos[2] = back_j[2] + data[2]
                next_j_pos[3] = back_j[3] + data[3]
                next_j_pos[4] = back_j[4] + data[4]
                next_j_pos[5] = back_j[5] + data[5]
            if distance <= end_pos_accuracy_mm:
                return action_array, j+1, next_j_pos
                print("Gotowe.Liczba kroków" + str(j))
            if j == 499:
                print("fail" + str(start_motion))
                return action_array, j, next_j_pos
            send1 = []
            send1.insert(0, back_j[0])
            send1.insert(1, back_j[1])
            send1.insert(2, back_j[2])
            send1.insert(3, back_j[3])
            send1.insert(4, back_j[4])
            send1.insert(5, back_j[5])
            send1.insert(6, next_j[0])
            send1.insert(7, next_j[1])
            send1.insert(8, next_j[2])
            send1.insert(9, next_j[3])
            send1.insert(10, next_j[4])
            send1.insert(11, next_j[5])
            setPos(send1, move_object_by_gripper)
            #time.sleep(0.5)
        back_j.insert(0,next_j_pos[0])
        back_j.insert(1,next_j_pos[1]) 
        back_j.insert(2,next_j_pos[2]) 
        back_j.insert(3,next_j_pos[3]) 
        back_j.insert(4,next_j_pos[4]) 
        back_j.insert(5,next_j_pos[5])

def AgentExecute(start_motion_local, robot, action_array, step, start_pos, move_object_by_gripper):
    #global start_motion, start_pos_of_motion, err11, err22, err33, err44, err55, err66
    back_j = []
    back_j.insert(0,start_pos[0])
    back_j.insert(1,start_pos[1]) 
    back_j.insert(2,start_pos[2]) 
    back_j.insert(3,start_pos[3]) 
    back_j.insert(4,start_pos[4]) 
    back_j.insert(5,start_pos[5])
    next_j = []
    next_j.insert(0, 0)
    next_j.insert(1, 0)
    next_j.insert(2, 0)
    next_j.insert(3, 0)
    next_j.insert(4, 0)
    next_j.insert(5, 0)
    for w in range(step):           
        data = list_of_action_six_axis[action_array[w]]
        send = []
        send.insert(0, back_j[0])
        send.insert(1, back_j[1])
        send.insert(2, back_j[2])
        send.insert(3, back_j[3])
        send.insert(4, back_j[4])
        send.insert(5, back_j[5])
        send.insert(6, data[0])
        send.insert(7, data[1])
        send.insert(8, data[2])
        send.insert(9, data[3])
        send.insert(10, data[4])
        send.insert(11, data[5])
        pos, pos1, pos2, pos3, pos4, pos5, joint1_range, joint2_range, joint3_range, joint4_range, joint5_range, joint6_range = setPosRender(send, move_object_by_gripper) 
        
        if (joint1_range == 0):
            data[0] = 0
                
        if (joint2_range == 0):
            data[1] = 0
                
        if (joint3_range == 0):
            data[2] = 0
                
        if (joint4_range == 0):
            data[3] = 0
                
        if (joint5_range == 0):
            data[4] = 0
                
        if (joint6_range == 0):
            data[5] = 0
               
        x_chart.append(pos[0])
        y_chart.append(pos[1])
        z_chart.append(pos[2])
        back_j[0] = back_j[0]+data[0]
        back_j[1] = back_j[1]+data[1]
        back_j[2] = back_j[2]+data[2]
        back_j[3] = back_j[3]+data[3]
        back_j[4] = back_j[4]+data[4]
        back_j[5] = back_j[5]+data[5]
        j1_rad.append(back_j[0])
        j2_rad.append(back_j[1])
        j3_rad.append(back_j[2])
        j4_rad.append(back_j[3])
        j5_rad.append(back_j[4])
        j6_rad.append(back_j[5]) 
        
        send1 = []
        send1.insert(0, back_j[0])
        send1.insert(1, back_j[1])
        send1.insert(2, back_j[2])
        send1.insert(3, back_j[3])
        send1.insert(4, back_j[4])
        send1.insert(5, back_j[5])
        send1.insert(6, next_j[0])
        send1.insert(7, next_j[1])
        send1.insert(8, next_j[2])
        send1.insert(9, next_j[3])
        send1.insert(10, next_j[4])
        send1.insert(11, next_j[5])
        setPosRender(send1, move_object_by_gripper)
        
        #Mitsu(robot, send1)#mitsubishi #robimy joint1,2,3,4,5,6 pobieramy do nich wartosci aktualne z robota i sprawadzamy roznice, jesli wieksza mniejsza niz zakladana to zmieniamy kinematyke od tego punktu
        '''
        joint_robot = []#ramka z robota o aktualnych pozycjach jointow
        joint_robot.insert(0, back_j[0])
        joint_robot.insert(1, back_j[1])
        joint_robot.insert(2, back_j[2])
        joint_robot.insert(3, back_j[3])
        joint_robot.insert(4, back_j[4])
        joint_robot.insert(5, back_j[5])
        '''
        '''
        err1 = abs(joint_robot[0] - back_j[0])
        err2 = abs(joint_robot[1] - back_j[1])
        err3 = abs(joint_robot[2] - back_j[2])
        err4 = abs(joint_robot[3] - back_j[3])
        err5 = abs(joint_robot[4] - back_j[4])
        err6 = abs(joint_robot[5] - back_j[5])

        if err1 > 0.001 or err2 > 0.001 or err3 > 0.001 or err4 > 0.001 or err5 > 0.001 or err6 > 0.001:
                     
            if err1 > 0.001:
                err11=0
            else:
                err11=1
            if err2 > 0.001:
                err22=0
            else:
                err22=1
            if err3 > 0.001:
                err33=0
            else:
                err33=1
            if err4 > 0.001:
                err44=0
            else:
                err44=1
            if err5 > 0.001:
                err55=0
            else:
                err55=1
            if err6 > 0.001:
                err66=0
            else:
                err66=1
                    
            start_motion = start_motion_local
            start_pos_of_motion = back_j
            TeachingTrajectory(start_motion_local, back_j, err11, err22, err33, err44, err55, err66)
            return True
        '''
    
    return False

#robot = MitsubishiPythonJointController(threshold_position=0.001, max_iteration_move=30000000)
#act_pos, status = robot.home()

def TeachingTrajectory(start_motion, start_pos_of_motion, joint1, joint2, joint3, joint4, joint5, joint6, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos, big_step, little_step):
    global steps1, steps2, steps3, steps4, steps5, steps6, steps7, action1, action2, action3, action4, action5, action6, action7, start_pos_first_motion, start_pos_second_motion, start_pos_third_motion, start_pos_fourth_motion, start_pos_fifth_motion, start_pos_sixth_motion, start_pos_seventh_motion
    if start_motion==0:
        start_pos_first_motion = start_pos_of_motion
    if start_motion==1:
        start_pos_second_motion = start_pos_of_motion
    if start_motion==2:
        start_pos_third_motion = start_pos_of_motion
    if start_motion==3:
        start_pos_fourth_motion = start_pos_of_motion
    if start_motion==4:
        start_pos_fifth_motion = start_pos_of_motion
    if start_motion==5:
        start_pos_sixth_motion = start_pos_of_motion
    if start_motion==6:
        start_pos_seventh_motion = start_pos_of_motion
    start = time.clock()
    if start_motion<1:
        ActionInit(big_step,  joint1, joint2, joint3, joint4, joint5, joint6)
        end_position = object_to_move_pos
        end_position[2] = end_position[2]
        print("End position1: " + str(end_position))
        action1, steps1, start_pos_second_motion = AgentTeaching(0, end_position, 20, 0.8, start_pos_first_motion, False, False, False, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos)
        print("Start position1: " + str(start_pos_second_motion))
    if start_motion<2:
        ActionInit(little_step, joint1, joint2, joint3, joint4, joint5, 1)
        end_position[2] = end_position[2]
        print("End position2: " + str(end_position))
        action2, steps2, start_pos_third_motion = AgentTeaching(1, end_position, 3.0, 20, start_pos_second_motion, False, True, False, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos)
        print("Start position2: " + str(start_pos_third_motion))
    if start_motion<3:
        ActionInit(big_step, joint1, joint2, joint3, 0, joint5, 0)
        end_position[2] = end_position[2]
        print("End position3: " + str(end_position))
        action3, steps3, start_pos_fourth_motion = AgentTeaching(2, end_position, 20, 0.8, start_pos_third_motion, True, False, False, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos)
        print("Start position3: " + str(start_pos_fourth_motion))
    if start_motion<4:
        ActionInit(big_step, joint1, joint2, joint3, 0, joint5, 0)
        end_position = hand_pos
        end_position[2] = end_position[2]
        print("End position4: " + str(end_position))
        action4, steps4, start_pos_fifth_motion = AgentTeaching(3, end_position, 20, 0.8, start_pos_fourth_motion, True, False, False, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos)
        print("Start position4: " + str(start_pos_fifth_motion))
    if start_motion<5:
        ActionInit(little_step, joint1, joint2, joint3, 0, joint5, 0)
        end_position[2] = end_position[2]
        print("End position5: " + str(end_position))
        action5, steps5, start_pos_sixth_motion = AgentTeaching(4, end_position, 3.0, 20, start_pos_fifth_motion, True, False, True, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos)
        print("Start position5: " + str(start_pos_sixth_motion))
    if start_motion<6:
        ActionInit(big_step, joint1, joint2, joint3, 0, joint5, 0)
        end_position = hand_pos
        end_position[2] = end_position[2]
        print("End position6: " + str(end_position))
        action6, steps6, start_pos_seventh_motion = AgentTeaching(5, end_position, 20, 0.8, start_pos_sixth_motion, False, False, False, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos)
        print("Start position6: " + str(start_pos_seventh_motion))
    if start_motion<7:
        ActionInit(big_step, joint1, joint2, joint3, 0, joint5, 0)
        end_position = np.array([820, 0, 813])
        action7, steps7, unknow = AgentTeaching(6, end_position, 20, 0.8, start_pos_seventh_motion, False, False, False, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos)
    end = time.clock()
    total = end - start

    print("{0:02f}s".format(total))


def ExecuteTrajectory(start_motion, start_pos_of_motion, joint1, joint2, joint3, joint4, joint5, joint6, big_step, little_step):
    global steps1, steps2, steps3, steps4, steps5, steps6, steps7, action1, action2, action3, action4, action5, action6, action7, start_pos_first_motion, start_pos_second_motion, start_pos_third_motion, start_pos_fourth_motion, start_pos_fifth_motion, start_pos_sixth_motion, start_pos_seventh_motion
    if start_motion==0:
        start_pos_first_motion = start_pos_of_motion
    if start_motion==1:
        start_pos_second_motion = start_pos_of_motion
    if start_motion==2:
        start_pos_third_motion = start_pos_of_motion
    if start_motion==3:
        start_pos_fourth_motion = start_pos_of_motion
    if start_motion==4:
        start_pos_fifth_motion = start_pos_of_motion
    if start_motion==5:
        start_pos_sixth_motion = start_pos_of_motion
    if start_motion==6:
        start_pos_seventh_motion = start_pos_of_motion
    robot = 0
    if start_motion < 1:
        ActionInit(big_step, joint1, joint2, joint3, joint4, joint5, joint6)
        stop = AgentExecute(0, robot, action1, steps1, start_pos_first_motion, False)
        if stop == True:
            return True
    if start_motion < 2:
        ActionInit(little_step, joint1, joint2, joint3, joint4, joint5, 1)
        stop = AgentExecute(1, robot, action2, steps2, start_pos_second_motion, False)
        if stop == True:
            return True
    if start_motion < 3:
        ActionInit(big_step, joint1, joint2, joint3, 0, joint5, 0)
        stop = AgentExecute(2, robot, action3, steps3, start_pos_third_motion, True)
        if stop == True:
            return True
    if start_motion < 4:
        ActionInit(big_step, joint1, joint2, joint3, 0, joint5, 0)
        stop = AgentExecute(3, robot, action4, steps4, start_pos_fourth_motion, True)
        if stop == True:
            return True
    if start_motion < 5:
        ActionInit(little_step, joint1, joint2, joint3, 0, joint5, 0)
        stop = AgentExecute(4, robot, action5, steps5, start_pos_fifth_motion, True)
        if stop == True:
            return True
    if start_motion < 6:
        ActionInit(big_step, joint1, joint2, joint3, 0, joint5, 0)
        stop = AgentExecute(5, robot, action6, steps6, start_pos_sixth_motion, False)
        if stop == True:
            return True
    if start_motion < 7:
        ActionInit(big_step, joint1, joint2, joint3, 0, joint5, 0)
        stop = AgentExecute(6, robot, action7, steps7, start_pos_seventh_motion, False)
        if stop == True:
            return True
        
def SavePosition():
    global x_chart, y_chart, z_chart, j1_rad, j2_rad, j3_rad, j4_rad, j5_rad, j6_rad    #/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/X_POS.txt
    
    file1 = open("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/X_POS.txt","w") 
    for items in x_chart:
        file1.write('%s\n' %items) 
    file1.close() 
    
    file2 = open("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/Y_POS.txt","w") 
    for items in y_chart:
        file2.write('%s\n' %items) 
    file2.close() 
    
    file3 = open("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/Z_POS.txt","w") 
    for items in z_chart:
        file3.write('%s\n' %items)  
    file3.close() 
    
    file4 = open("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/j1_rad.txt","w") 
    for items in j1_rad:
        file4.write('%s\n' %items) 
    file4.close() 
    
    file5 = open("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/j2_rad.txt","w") 
    for items in j2_rad:
        file5.write('%s\n' %items) 
    file5.close() 
    
    file6 = open("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/j3_rad.txt","w") 
    for items in j3_rad:
        file6.write('%s\n' %items)
    file6.close() 
    
    file7 = open("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/j4_rad.txt","w") 
    for items in j4_rad:
        file7.write('%s\n' %items) 
    file7.close() 
    
    file8 = open("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/j5_rad.txt","w") 
    for items in j5_rad:
        file8.write('%s\n' %items) 
    file8.close() 
    
    file9 = open("/home/patryk/anaconda3/envs/AI_2020/myFile/mitsubishi_mujoco/j6_rad.txt","w") 
    for items in j6_rad:
        file9.write('%s\n' %items) 
    file9.close() 


def SearchingTrajectory(start_motion, start_pos_of_motion, joint1, joint2, joint3, joint4, joint5, joint6, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos, log_file, big_step, little_step):
    global steps1, steps2, steps3, steps4, steps5, steps6, steps7, action1, action2, action3, action4, action5, action6, action7, start_pos_first_motion, start_pos_second_motion, start_pos_third_motion, start_pos_fourth_motion, start_pos_fifth_motion, start_pos_sixth_motion, start_pos_seventh_motion
    start = time.clock()
    backup_pos_1 = []
    backup_pos_2 = []
    backup_pos_3 = []
    backup_pos_4 = []
    backup_pos_5 = []
    backup_pos_6 = []
    backup_pos_7 = []
    backup_action1 = []
    backup_action2 = []
    backup_action3 = []
    backup_action4 = []
    backup_action5 = []
    backup_action6 = []
    backup_action7 = []
    backup_steps1 = 0
    backup_steps2 = 0
    backup_steps3 = 0
    backup_steps4 = 0
    backup_steps5 = 0
    backup_steps6 = 0
    backup_steps7 = 0
    
    return_param1 = 0
    return_param2 = 0
    myGenetic = genetic(joint1, joint2, joint3, joint4, joint5, joint6, big_step/1000, little_step/1000, object_to_move_pos/1000, hand_pos/1000, enemy1_pos/1000, enemy2_pos/1000)
    last_best_steps = 1000
    motion_success = False
    #kolejny for z liczba populacji na start maks 10
    #inicjalizacja klasy Genetic i inicjalizacja pierwszej populacji
    my_list_of_population = []
    my_array_of_population = []
    my_list_of_population = myGenetic.GenerateFirstPopulation()
    for epo in range(3):       
        my_array_of_population = np.array(my_list_of_population)
        for i in range(20):                                            #2 parametry po 10 razem 100 kombinacji
            #pobieramy parametry p1 i p2 z genetica tam musi byc tablica 100x2
            param1 = my_array_of_population[i][0]
            param2 = my_array_of_population[i][1]
            start_pos_first_motion = start_pos_of_motion
            my_move = 0
            '''
            if start_motion==0:
                start_pos_first_motion = start_pos_of_motion
            if start_motion==1:
                start_pos_second_motion = start_pos_of_motion
            if start_motion==2:
                start_pos_third_motion = start_pos_of_motion
            if start_motion==3:
                start_pos_fourth_motion = start_pos_of_motion
            if start_motion==4:
                start_pos_fifth_motion = start_pos_of_motion
            if start_motion==5:
                start_pos_sixth_motion = start_pos_of_motion
            if start_motion==6:
                start_pos_seventh_motion = start_pos_of_motion
            '''  
            if start_motion<1:
                my_move = 1
                ActionInit(big_step,  joint1, joint2, joint3, joint4, joint5, joint6)
                end_position = np.array([object_to_move_pos[0], object_to_move_pos[1], object_to_move_pos[2]])
                action1, steps1, start_pos_second_motion, motion_success = AgentSearching(param1, param2, 0, end_position, 20, 0.8, start_pos_first_motion, False, False, False, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos)
            if start_motion<2 and motion_success == True:
                my_move = 2
                ActionInit(little_step, joint1, joint2, joint3, joint4, joint5, joint6)
                end_position = np.array([object_to_move_pos[0], object_to_move_pos[1], object_to_move_pos[2]])
                action2, steps2, start_pos_third_motion, motion_success = AgentSearching(param1, param2, 1, end_position, 5, 20, start_pos_second_motion, False, True, False, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos)
            if start_motion<3 and motion_success == True:
                my_move = 3
                ActionInit(big_step, joint1, joint2, joint3, joint4, joint5, joint6)
                end_position = np.array([object_to_move_pos[0], object_to_move_pos[1], object_to_move_pos[2]])
                action3, steps3, start_pos_fourth_motion, motion_success = AgentSearching(param1, param2, 2, end_position, 20, 0.8, start_pos_third_motion, True, False, False, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos)
            if start_motion<4 and motion_success == True:
                my_move = 4
                ActionInit(big_step, joint1, joint2, joint3, joint4, joint5, joint6)
                end_position = np.array([hand_pos[0], hand_pos[1], hand_pos[2]])
                action4, steps4, start_pos_fifth_motion, motion_success = AgentSearching(param1, param2, 3, end_position, 20, 0.8, start_pos_fourth_motion, True, False, False, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos)
            if start_motion<5 and motion_success == True:
                my_move = 5
                ActionInit(little_step, joint1, joint2, joint3, joint4, joint5, joint6)
                end_position = np.array([hand_pos[0], hand_pos[1], hand_pos[2]])
                action5, steps5, start_pos_sixth_motion, motion_success = AgentSearching(param1, param2, 4, end_position, 5, 20, start_pos_fifth_motion, True, False, True, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos)
            if start_motion<6 and motion_success == True:
                my_move = 6
                ActionInit(big_step, joint1, joint2, joint3, joint4, joint5, joint6)
                end_position = np.array([hand_pos[0], hand_pos[1], hand_pos[2]])
                action6, steps6, start_pos_seventh_motion, motion_success = AgentSearching(param1, param2, 5, end_position, 20, 0.8, start_pos_sixth_motion, False, False, False, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos)
            if start_motion<7 and motion_success == True:
                my_move = 7
                ActionInit(big_step, joint1, joint2, joint3, joint4, joint5, joint6)
                end_position = np.array([820, 0, 813])
                action7, steps7, unknow, motion_success = AgentSearching(param1, param2, 6, end_position, 20, 0.8, start_pos_seventh_motion, False, False, False, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos)
            
            all_steps = steps1 + steps2 + steps3 + steps4 + steps5 + steps6 + steps7
            my_list_of_population.append(all_steps)
            
            if all_steps < last_best_steps and motion_success == True:
                last_best_steps = all_steps
                return_param1 = param1
                return_param2 = param2
                best = []
                best.clear()
                best.append(return_param1)
                best.append(return_param2)
                myGenetic.SetBestIndividual(best)
                backup_pos_1 = start_pos_first_motion
                backup_pos_2 = start_pos_second_motion
                backup_pos_3 = start_pos_third_motion
                backup_pos_4 = start_pos_fourth_motion
                backup_pos_5 = start_pos_fifth_motion
                backup_pos_6 = start_pos_sixth_motion
                backup_pos_7 = start_pos_seventh_motion
                backup_action1 = action1
                backup_action2 = action2
                backup_action3 = action3
                backup_action4 = action4
                backup_action5 = action5
                backup_action6 = action6
                backup_action7 = action7
                backup_steps1 = steps1
                backup_steps2 = steps2
                backup_steps3 = steps3
                backup_steps4 = steps4
                backup_steps5 = steps5
                backup_steps6 = steps6
                backup_steps7 = steps7
            next_line="\n"
            print("Best reward function for motion: " + str(all_steps) + " iteration: " + str(i) + " param1: " + str(param1) + " param2: " + str(param2))
            print("Motion: " + str(my_move))
            log_file.writelines("Best reward function for motion: " + str(all_steps) + " iteration: " + str(i) + " param1: " + str(param1) + " param2: " + str(param2))
            log_file.writelines(next_line)
            log_file.writelines("Motion: " + str(my_move))
            log_file.writelines(next_line)
        #po wyjsciu z iteraci 100 kombinacji sprawdzamy czy mamy spelniony warunek motion success jesli tak nie robimy kolejnej populacji i konczymy jesli nie krzyzujemy i mutujemy 10 najlepszych wynikow
        best = myGenetic.GetBestIndividual(my_list_of_population)
        cross = myGenetic.CrossingBestIndividual(best)
        my_list_of_population.clear()
        my_list_of_population = myGenetic.Mutation(cross)
        end = time.clock()
        total = end - start
    
    start_pos_first_motion = backup_pos_1
    start_pos_second_motion = backup_pos_2
    start_pos_third_motion = backup_pos_3
    start_pos_fourth_motion = backup_pos_4
    start_pos_fifth_motion = backup_pos_5
    start_pos_sixth_motion = backup_pos_6
    start_pos_seventh_motion = backup_pos_7
    action1 = backup_action1
    action2 = backup_action2
    action3 = backup_action3
    action4 = backup_action4
    action5 = backup_action5
    action6 = backup_action6
    action7 = backup_action7
    steps1 = backup_steps1
    steps2 = backup_steps2
    steps3 = backup_steps3
    steps4 = backup_steps4
    steps5 = backup_steps5
    steps6 = backup_steps6
    steps7 = backup_steps7

    print("Best param: " + str(return_param1) + "   " + str(return_param2))
    print("{0:02f}s".format(total))
    log_file.writelines("Best param: " + str(return_param1) + "   " + str(return_param2))
    log_file.writelines(next_line)
    log_file.writelines("{0:02f}s".format(total))
    log_file.writelines(next_line)
    try:
        log_file.writelines("Save param: " + str(return_param1) + "   " + str(return_param2))
        if return_param1 != 0.0 and return_param2 != 0.0:
            myGenetic.SaveDataToTxt()
    except Exception:
        log_file.writelines("Not save param: " + str(return_param1) + "   " + str(return_param2))
        
    return return_param1, return_param2
    
def ObjectDetectionAndLocation():
    global enemy1_pos, enemy2_pos, hand_pos, object_to_move_pos
    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    # Import utilites
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'doctoral_GRAPH'
    IMAGE_NAME = 'All_86.jpg'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'doctoral_training','object-detection.pbtxt')

    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

    # Number of classes the object detector can identify
    NUM_CLASSES = 3

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    images,  my_dict= vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.55)

    key_list = list(my_dict.keys())
    key_array = np.array(key_list)
    val_list = list(my_dict.values())
    enemy1_x = 5000 
    enemy1_y = 5000
    enemy2_x = 5000
    enemy2_y = 5000
    multimeter_x = 5000
    multimeter_y = 5000
    hand_x = 5000
    hand_y = 5000
    for i in range(len(val_list)):
        substring_in_list = any("enemy_top" in string for string in val_list[i])
        if substring_in_list==True:
            if enemy1_x == 5000 and enemy1_y == 5000:
                enemy1_x = (key_array[i][3]*1280+key_array[i][1]*1280)/2
                enemy1_y = (key_array[i][2]*720+key_array[i][0]*720)/2
            else:
                enemy2_x = (key_array[i][3]*1280+key_array[i][1]*1280)/2
                enemy2_y = (key_array[i][2]*720+key_array[i][0]*720)/2
        substring_in_list = any("multimeter" in string for string in val_list[i])
        if substring_in_list==True:
            multimeter_x = (key_array[i][3]*1280+key_array[i][1]*1280)/2
            multimeter_y = (key_array[i][2]*720+key_array[i][0]*720)/2
        substring_in_list = any("hand" in string for string in val_list[i])
        if substring_in_list==True:
            hand_x = (key_array[i][3]*1280+key_array[i][1]*1280)/2
            hand_y = (key_array[i][2]*720+key_array[i][0]*720)/2

        
    
    image = cv2.circle(image, (int(enemy1_x), int(enemy1_y)), 4, (0, 255, 0), -1)
    image = cv2.circle(image, (int(enemy2_x), int(enemy2_y)), 4, (0, 255, 0), -1)
    image = cv2.circle(image, (int(multimeter_x), int(multimeter_y)), 4, (0, 255, 0), -1)
    image = cv2.circle(image, (int(hand_x), int(hand_y)), 4, (0, 255, 0), -1)
    
    #dodane bedzie obliczanie wspolrzednych
    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)

    # Press any key to close the image
    cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()
    
    if TEST == True:       
        object_to_move_pos = object_to_move_pos
        hand_pos = hand_pos
        enemy1_pos = enemy1_pos
        enemy2_pos = enemy2_pos
    
    return object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos
    
param1 = 0
param2 = 0

def Generate_random_situation():
    axis1 = 1
    axis2 = 1
    axis3 = 1
    axis4 = 1
    axis5 = 1
    axis6 = 1
    
    deactivate_axis1 = random.randint(1, 7)
    #deactivate_axis2 = random.randint(1, 8)
    if deactivate_axis1 == 1:# or deactivate_axis2 == 1:
        axis1 = 0
    if deactivate_axis1 == 2:#  or deactivate_axis2 == 2:
        axis2 = 0
    if deactivate_axis1 == 3:#  or deactivate_axis2 == 3:
        axis3 = 0
    if deactivate_axis1 == 4:#  or deactivate_axis2 == 4:
        axis4 = 0
    if deactivate_axis1 == 5:#  or deactivate_axis2 == 5:
        axis5 = 0
    if deactivate_axis1 == 6:#  or deactivate_axis2 == 6:
        axis6 = 0

     
    pos_object = []
    pos_end = []
    pos_enemy1 = []
    pos_enemy2 = []
    
    x_pos_end = 0
    y_pos_end = 0
    z_pos_end = 0
    x_pos_enemy1 = 0
    y_pos_enemy1 = 0
    z_pos_enemy1 = 0
    x_pos_enemy2 = 0
    y_pos_enemy2 = 0
    z_pos_enemy2 = 0
    
    x_pos_enemy1 = random.randint(650, 950)
    y_pos_enemy1 = random.randint(-150, 150)
    z_pos_enemy1 = 700
    
    distance1 = 0
    distance2 = 0
    distance3 = 0
    distance4 = 0
    distance5 = 0
    distance6 = 0
    while distance1 < 350 or distance2 < 200 or distance3 < 200 or distance4 < 150 or distance5 < 200 or distance6 < 200:
        x_pos_object = random.randint(600, 1000)
        y_pos_object = random.randint(-250, 250)
        z_pos_object = random.randint(400, 800)
        x_pos_end = random.randint(600, 1000)
        y_pos_end = random.randint(-250, 250)
        z_pos_end = random.randint(400, 800) 
        x_pos_enemy2 = random.randint(650, 950)
        y_pos_enemy2 = random.randint(-150, 150)
        z_pos_enemy2 = 600
        distance1 = math.sqrt((x_pos_object-x_pos_end)**2 + (y_pos_object-y_pos_end)**2)
        distance2 = math.sqrt((x_pos_object-x_pos_enemy1)**2 + (y_pos_object-y_pos_enemy1)**2)
        distance3 = math.sqrt((x_pos_end-x_pos_enemy1)**2 + (y_pos_end-y_pos_enemy1)**2)
        
        distance4 = math.sqrt((x_pos_enemy2-x_pos_enemy1)**2 + (y_pos_enemy2-y_pos_enemy1)**2)
        distance5 = math.sqrt((x_pos_object-x_pos_enemy2)**2 + (y_pos_object-y_pos_enemy2)**2)
        distance6 = math.sqrt((x_pos_end-x_pos_enemy2)**2 + (y_pos_end-y_pos_enemy2)**2)

    '''   
    distance1 = 0
    distance2 = 0
    distance3 = 0
    while (distance1 < 230) or (distance2 < 230) or (distance3 < 200):
        x_pos_enemy2 = random.randint(600, 1000)
        y_pos_enemy2 = random.randint(-200, 200)
        z_pos_enemy2 = 450
        distance1 = math.sqrt((x_pos_object-x_pos_enemy2)**2 + (y_pos_object-y_pos_enemy2)**2)
        distance2 = math.sqrt((x_pos_end-x_pos_enemy2)**2 + (y_pos_end-y_pos_enemy2)**2)
        distance3 = math.sqrt((x_pos_enemy1-x_pos_enemy2)**2 + (y_pos_enemy1-y_pos_enemy2)**2)
    '''
    pos_object = np.array([x_pos_object, y_pos_object, z_pos_object])
    pos_end = np.array([x_pos_end, y_pos_end, z_pos_end])
    pos_enemy1 = np.array([x_pos_enemy1, y_pos_enemy1, z_pos_enemy1])
    pos_enemy2 = np.array([x_pos_enemy2, y_pos_enemy2, z_pos_enemy2]) #pos_enemy1#
    big_step = random.uniform(5.0, 10.0)
    little_step = random.uniform(0.5, 2.0)
    
    return axis1, axis2, axis3, axis4, axis5, axis6, pos_object, pos_end, pos_enemy1, pos_enemy2, big_step, little_step
    
    
class MainApp(App):
    def build(self):
        start=Widget()
        
        buttonTeach = Button(text='Teaching', on_press=self.btnTeachCallback)#, on_press=self.btnTeachCallback) #(text='Teaching', size_hint=(0.2, 0.2), font size='20sp', pos_hint={'center_x':0.5, 'center_y':0.5}, on_press=self.pressBtn)
        buttonExecute = Button(text='Executing', on_press=self.btnExecuteCallback)
        buttonSaveTrajectory = Button(text='    Save \nTrajectory', on_press=self.btnSaveTrajectoryCallback)
        buttonSerachRewardFunction = Button(text='Search', on_press=self.btnSearchCallback)
        
        start.add_widget(buttonTeach)
        start.add_widget(buttonExecute)
        start.add_widget(buttonSaveTrajectory)
        start.add_widget(buttonSerachRewardFunction)
        
        buttonTeach.center = (70, 470)
        buttonExecute.center = (70, 350)
        buttonSaveTrajectory.center = (70, 230)
        buttonSerachRewardFunction.center = (70, 110)
        
        return start
      
    def btnTeachCallback(self, obj): 
        global start_pos_of_motion, err11, err22, err33, err44, err55, err66, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos
        print("Teach")
        #ObjectDetectionAndLocation()
        start_pos_of_motion = []
        start_pos_of_motion.insert(0, 0)
        start_pos_of_motion.insert(1, 0)
        start_pos_of_motion.insert(2, 0)
        start_pos_of_motion.insert(3, 0)
        start_pos_of_motion.insert(4, 0)
        start_pos_of_motion.insert(5, 0)
        print(start_pos_of_motion)
        
        big_step = big_step_main
        little_step = little_step_main
        
        TeachingTrajectory(0, start_pos_of_motion, err11, err22, err33, err44, err55, err66, enemy1_pos, enemy2_pos, object_to_move_pos, hand_pos, big_step, little_step)
        
    def btnExecuteCallback(self, obj):
        global viewer, sim, start_motion, start_pos_of_motion, err11, err22, err33, err44, err55, err66
        start_pos_of_motion = []
        start_pos_of_motion.insert(0, 0)
        start_pos_of_motion.insert(1, 0)
        start_pos_of_motion.insert(2, 0)
        start_pos_of_motion.insert(3, 0)
        start_pos_of_motion.insert(4, 0)
        start_pos_of_motion.insert(5, 0)
        viewer = MjViewer(sim)
        print("Execute")
        
        big_step = big_step_main
        little_step = little_step_main
        
        stop = ExecuteTrajectory(0, start_pos_of_motion, err11, err22, err33, err44, err55, err66, big_step, little_step)
        
        if stop == True:
            print("STOPPPPPPPP")
            ExecuteTrajectory(start_motion, start_pos_of_motion, err11, err22, err33, err44, err55, err66)
        window = glfw.get_current_context()
        
        time.sleep(2)
        glfw.destroy_window(window)
        
    def btnSaveTrajectoryCallback(self, obj):
        SavePosition()
        
    def btnSearchCallback(self, obj):
        print("search")
        global start_pos_of_motion, err11, err22, err33, err44, err55, err66, param1, param2
        #object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos = ObjectDetectionAndLocation()
        start_pos_of_motion = []
        start_pos_of_motion.insert(0, 0)
        start_pos_of_motion.insert(1, 0)
        start_pos_of_motion.insert(2, 0)
        start_pos_of_motion.insert(3, 0)
        start_pos_of_motion.insert(4, 0)
        start_pos_of_motion.insert(5, 0)
        #print(str(object_to_move_pos))
        log_file = open("Genetic_Log.txt", "w")
        next_line="\n"
        for i in range(30):
            try:
                print("Situation number: " + str(i+1))
                log_file.writelines("Situation number: " + str(i+1))
                log_file.writelines(next_line)
                err11, err22, err33, err44, err55, err66, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos, big_step, little_step = Generate_random_situation()
                print("Situation: " + str(err11) + " " + str(err22) + " " + str(err33) + " " + str(err44) + " " + str(err55) + " " + str(err66) + " " + str(big_step) + " " + str(little_step) + " "+ str(object_to_move_pos) + " " + str(hand_pos) + " " + str(enemy1_pos) + " " + str(enemy2_pos))
                log_file.writelines("Situation: " + str(err11) + " " + str(err22) + " " + str(err33) + " " + str(err44) + " " + str(err55) + " " + str(err66) + " " + str(big_step) + " " + str(little_step) + " " + str(object_to_move_pos) + " " + str(hand_pos) + " " + str(enemy1_pos) + " " + str(enemy2_pos))
                log_file.writelines(next_line)
                param1, param2 = SearchingTrajectory(0, start_pos_of_motion, err11, err22, err33, err44, err55, err66, object_to_move_pos, hand_pos, enemy1_pos, enemy2_pos, log_file, big_step, little_step)
            except Exception:
                print("ERROR DONT FIND PARAM")
                log_file.close()
        log_file.close()
  
if __name__ == "__main__":
    MainApp().run()   
    