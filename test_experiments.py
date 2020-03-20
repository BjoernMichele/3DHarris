import networkx as nx
from scipy.spatial import Delaunay
import numpy as np
import base 
import ply
import copy
from scipy.spatial.transform import Rotation as R


def dummy_function(mesh_object):
    return list([[1,2,3], [4,5,6]])

def dummy_function_transform(mesh_object):
    return list([[1,2,3], [4,5,6]])


def roation_transformation(mesh_object, angle): 
    #angle = [alpha, beta, gamma]
    #Rotation Transformation 
    #Apply Rotation 
    #Only work on a copy of the original 
    mesh_object = copy.deepcopy(mesh_object)
    rotations = angle
    inverse_rotations = [-rotations[2],-rotations[1],-rotations[0]]
    r = R.from_euler('xyz',rotations, degrees=True)
    r_inverse = R.from_euler('zyx', inverse_rotations, degrees=True)

    rotated_mesh =  base.Meshgrid(r.apply(mesh_object.np_position_array)) #Frage ob man auch den Graphen neu berechnen sollte, eigt ja schon
    interest_points = dummy_function_transform(rotated_mesh)
    interest_points_original = r_inverse.apply(interest_points)
    return interest_points_original


def scale_transformation(mesh_object, scale_factor): 
    #angle = [alpha, beta, gamma]
    #Rotation Transformation 
    #Apply Scaling 
    #Only work on a copy of the original 
    mesh_object = copy.deepcopy(mesh_object)
    scaled_mesh = base.Meshgrid(scale_factor * mesh_object.np_position_array)
    interest_points_scaled = dummy_function(scaled_mesh)
    interest_points_original = interest_points_scaled / scale_factor

    return interest_points_original


def calc_repeatability(ori_ip, trans_ip, threshold): 
    #Calculates the repeatability of the transformed ip and the original ip
    if type(ori_ip) is list: 
        ori_ip = np.array(ori_ip)
    if type(trans_ip) is list: 
        trans_ip = np.array(trans_ip) 

    ori_rep = np.repeat(ori_ip,trans_ip.shape[0],axis=0)
    trans_rep = np.tile(trans_ip,(ori_ip.shape[0],1))
    diff = np.linalg.norm(ori_rep -trans_rep,axis=1)

    bool_mask = np.any(np.reshape(diff<threshold, (-1,trans_ip.shape[0])),axis = 1) 
    repeatbility = float(np.sum(bool_mask)) / float(ori_ip.shape[0])
    return repeatbility

def test_pipeline(path_file):
    data = ply.read_ply(path_file)
    pos = np.stack([data['x'],data['y'], data['z']]).T #Positions: Nx3
    mesh_objects =base.Meshgrid(pos)
    
    #Original interest points
    interest_points = dummy_function(mesh_objects)
    #
    angle = np.array([0,45,0])
    interest_points_rotated = roation_transformation(mesh_objects, angle)
    #Calculate repeatability
    repeatability_value = calc_repeatability(interest_points, interest_points_rotated,2.0)
    print(repeatability_value)

    scale_factor = 1.1
    interest_points_scales = scale_transformation(mesh_objects,scale_factor)
    repeatability_value_scale = 


if __name__ == "__main__":
    test_pipeline("bunny.ply")