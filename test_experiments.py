import networkx as nx
import numpy as np
import base
import ply
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from numba import jit
import numba
import os 

class Experiment: 
    def __init__(self,path_file, sel_mod = "rel", self_args = {'thresh': 0.01},\
         neigh_args = {'k':10}, neigh_flag = "k", k_harris = 0.04): 
        data = ply.read_ply(path_file)
        pos = np.stack([data['x'],data['y'], data['z']]).T #Positions: Nx3
        self.mesh_objects =base.Meshgrid(pos)
        self.repeatbiliy_thresh = self.mesh_objects.diameter * 0.01
        self.sel_mod = sel_mod
        self.sel_args = self_args
        self.neigh_args = neigh_args
        self.neigh_flag = neigh_flag 
        self.k_harris = k_harris 
        assert not np.all(np.isnan(pos)) and not np.all(np.isinf(pos))

    def roation_transformation(self, mesh_object, angle,write_ply = False, path = "base"):
        #angle = [alpha, beta, gamma]
        #Rotation Transformation
        #Apply Rotation
        #Only work on a copy of the original
        mesh_object = copy.deepcopy(mesh_object)
        rotations = angle
        inverse_rotations = [-rotations[2],-rotations[1],-rotations[0]]
        r = R.from_euler('xyz',rotations, degrees=True)
        r_inverse = R.from_euler('zyx', inverse_rotations, degrees=True)

        rotated_mesh =  base.Meshgrid(r.apply(mesh_object.points)) #Frage ob man auch den Graphen neu berechnen sollte, eigt ja schon
        interest_points = rotated_mesh.get_interest_points(sel_mod =self.sel_mod,\
            sel_args = self.sel_args,neigh_flag = self.neigh_flag, neigh_args = self.neigh_args,\
                 k_harris = self.k_harris)
        if write_ply: 
            ply.write_ply(path + "_all", rotated_mesh.points, ['x','y','z'])
        if write_ply: 
            ply.write_ply(path + "_int", interest_points, ['x','y','z'])
        
        
        interest_points_original = r_inverse.apply(interest_points)
        return interest_points_original

    def scale_transformation(self, mesh_object, scale_factor,write_ply = False, path = "base"):
        #angle = [alpha, beta, gamma]
        #Rotation Transformation
        #Apply Scaling
        #Only work on a copy of the original
        mesh_object = copy.deepcopy(mesh_object)
        scaled_mesh = base.Meshgrid(scale_factor * mesh_object.points)
        interest_points_scaled = scaled_mesh.get_interest_points(sel_mod =self.sel_mod,\
            sel_args = self.sel_args,neigh_flag = self.neigh_flag, neigh_args = self.neigh_args,\
                 k_harris = self.k_harris)
        if write_ply: 
            ply.write_ply(path + "_all", scaled_mesh.points, ['x','y','z'])
        if write_ply: 
            ply.write_ply(path + "_int", interest_points_scaled, ['x','y','z'])
        interest_points_original = interest_points_scaled / scale_factor

        return interest_points_original
    def gaus_noise_transformation(self, mesh_object, noise_level, absolute_noise_factor,write_ply = False, path = "base"):
        #noise level: std of the gaussian additive noise
        #absolute noise factor --> to make the absolute value of the noise in the range of  the size 
        #--> factor which is multiplied with the diameter of the object
        
        mesh_object = copy.deepcopy(mesh_object)

        noise_x = np.random.normal(0,noise_level,mesh_object.points.shape[0]) * mesh_object.width * absolute_noise_factor
        noise_y = np.random.normal(0,noise_level,mesh_object.points.shape[0]) * mesh_object.height * absolute_noise_factor
        noise_z = np.random.normal(0,noise_level,mesh_object.points.shape[0]) * mesh_object.depth * absolute_noise_factor
        
        noise = np.stack([noise_x,noise_y,noise_z]).T
        noised_points = mesh_object.points  + noise
        if write_ply: 
            ply.write_ply(path + "_all", noised_points, ['x','y','z'])
        noise_mesh = base.Meshgrid(noised_points)
        interest_points_original = noise_mesh.get_interest_points(sel_mod =self.sel_mod,\
            sel_args = self.sel_args,neigh_flag = self.neigh_flag, neigh_args = self.neigh_args,\
                 k_harris = self.k_harris)

        if write_ply: 
            ply.write_ply(path + "_ip", interest_points_original, ['x','y','z'])

        return interest_points_original
    def translation_transformation(self, mesh_object, translation,write_ply = False, path = "base"):
        #angle = [alpha, beta, gamma]
        #Rotation Transformation
        #Apply Scaling
        #Only work on a copy of the original
        mesh_object = copy.deepcopy(mesh_object)
        translated_mesh = base.Meshgrid(translation + mesh_object.points)
        interest_points_transl = translated_mesh.get_interest_points(sel_mod =self.sel_mod,\
            sel_args = self.sel_args,neigh_flag = self.neigh_flag, neigh_args = self.neigh_args,\
                 k_harris = self.k_harris)
        
        if write_ply: 
            ply.write_ply(path + "_all", translated_mesh.points, ['x','y','z'])
        if write_ply: 
            ply.write_ply(path + "_int", interest_points_transl, ['x','y','z'])
        
        
        interest_points_original = interest_points_transl - translation

        return interest_points_original
    
    def local_holes_transformation(self, mesh_object, nb_holes, hole_size,write_ply = False, path = "base"): 
        mesh_object = copy.deepcopy(mesh_object)
        reject_groups = np.random.randint(mesh_object.points.shape[0], size=nb_holes)
        tree = KDTree(mesh_object.points, leaf_size=2)
        complete_rejected_indices = []
        for reject_index in reject_groups: 
            indices = tree.query_radius(mesh_object.points[reject_index].reshape(1,-1), r=hole_size)[0]
            
            complete_rejected_indices.extend(indices.tolist())
            complete_rejected_indices.append(reject_index.tolist())
            
        
        delete_index = list(set(complete_rejected_indices))
        
        new_points = np.delete(mesh_object.points, delete_index, axis = 0)
        print("After deletion, nb_points remaining : {}".format(new_points.shape))
        
        holes_mesh = base.Meshgrid(new_points)

        interest_points_holes = holes_mesh.get_interest_points(sel_mod =self.sel_mod,\
            sel_args = self.sel_args,neigh_flag = self.neigh_flag, neigh_args = self.neigh_args,\
                 k_harris = self.k_harris)
        
        if write_ply: 
            ply.write_ply(path + "_all", holes_mesh.points, ['x','y','z'])
        if write_ply: 
            ply.write_ply(path + "_int", interest_points_holes, ['x','y','z'])
        
        
        return interest_points_holes
        
    def grid_subsampling_transformation(self, mesh_object, grid_size,write_ply = False, path = "base"): 
        mesh_object = copy.deepcopy(mesh_object)
        subsampled_points = self.grid_subsampling(mesh_object.points, grid_size)
        subsampled_mesh = base.Meshgrid(subsampled_points)
        interest_points_original = subsampled_mesh.get_interest_points(sel_mod =self.sel_mod,\
            sel_args = self.sel_args,neigh_flag = self.neigh_flag, neigh_args = self.neigh_args,\
                 k_harris = self.k_harris)

        if write_ply: 
            ply.write_ply(path + "_all", subsampled_mesh.points, ['x','y','z'])
        if write_ply: 
            ply.write_ply(path + "_int", interest_points_original, ['x','y','z'])
        return interest_points_original

    def grid_subsampling(self, points, voxel_size):
        print("Number points {}".format(points.shape))
        indices = (points/voxel_size+0.5).astype(np.int32)
        pos_dict = {}
        count_dict = {}
        for i, index in enumerate(indices): 
            index_key = (index[0],index[1],index[2])
            if index_key in pos_dict: 
                pos_dict[index_key] += points[i]
                count_dict[index_key] +=1
            else:
                pos_dict[index_key] = points[i]
                count_dict[index_key] = 1
        position_list = []
        print("Number voxels {}".format(len(pos_dict.keys())))
        for key, values in pos_dict.items():
            position = values / count_dict[key]
            position_list.append(position)
        subsampled_points = np.array(position_list)

        return subsampled_points
    
    def calc_repeatability_reverse(self, ori_ip, trans_ip, threshold): 
        #Calculates the repeatability in both ways
        rep1 = self.calc_repeatability(ori_ip, trans_ip, threshold)
        rep2 = self.calc_repeatability(trans_ip, ori_ip, threshold)
        return (rep1 + rep2) / 2.0

    def calc_repeatability(self, ori_ip, trans_ip, threshold):
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

    def test_pipeline(self, write_ply = False, path = "base"):
        #Original interest points
        interest_points =  self.mesh_objects.get_interest_points(sel_mod =self.sel_mod,\
            sel_args = self.sel_args,neigh_flag = self.neigh_flag, neigh_args = self.neigh_args,\
                 k_harris = self.k_harris)

        if write_ply: 
            path_all = os.path.join(path, "base_all")
            path_tmp = os.path.join(path, "base_int")

            ply.write_ply(path_all, self.mesh_objects.points, ['x','y', 'z'])
            ply.write_ply(path_tmp, interest_points, ['x','y', 'z'])
        #Angle repeatability
        path_angle = os.path.join(path,"angle/")
        print("Path angle {}".format(path_angle))
        angle_repeat_list = []
        angles = [[0,1,0],[0,5,0], [0,10,0], [0,15,0],[0,20,0],[0,30,0],[0,45,0],[0,75,0],[0,90,0],[0,120,0],[0,150,0],[0,180,0]]
        for angle in angles:
            angle = np.array(angle)
            path_angle_tmp = path_angle + str(angle[1])
            interest_points_rotated = self.roation_transformation(self.mesh_objects, angle,write_ply = write_ply, path=path_angle_tmp)
            #Calculate repeatability
            
            repeatability_value = self.calc_repeatability_reverse(interest_points, interest_points_rotated,self.repeatbiliy_thresh)
            angle_repeat_list.append(repeatability_value)
            print("Rep value angle{}".format(repeatability_value))

        scale_list = []
        scale_factors = [0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
        path_scale = os.path.join(path,"scale/")
        for scale_factor in scale_factors:
            path_scale_tmp = path_scale + str(scale_factor)
            interest_points_scales = self.scale_transformation(self.mesh_objects,scale_factor,write_ply = write_ply, path= path_scale_tmp)
            repeatability_value_scale = self.calc_repeatability_reverse(interest_points, interest_points_scales,self.repeatbiliy_thresh)
            print("Rep value scale{}".format(repeatability_value_scale))
            scale_list.append(repeatability_value_scale)

        translation = np.array([10.0, 20.0, -15.0])
        path_transl = os.path.join(path,"translation/")
        interest_points_translated = self.translation_transformation(self.mesh_objects, translation,write_ply= write_ply, path= path_transl)
        repeatability_value_transl = self.calc_repeatability_reverse(interest_points, interest_points_translated,self.repeatbiliy_thresh)
        print("Rep. value transl {}".format(repeatability_value_transl))

        #Subsampling (grid resolution)
        grid_resolutions = [0.0001, 0.001,0.0025, 0.005, 0.0075, 0.01,0.05, 0.1]
        subsampling_list = []
        path_subs = os.path.join(path,"subsampling/")
        for grid_resolution in grid_resolutions:
            path_subs_tmp = path_subs + str(grid_resolution)

            grid_size = self.mesh_objects.diameter * grid_resolution
            interest_points_subsampled = self.grid_subsampling_transformation( self.mesh_objects, grid_size, write_ply=write_ply, path=path_subs_tmp)
            
            repeatability_value_subsampled = self.calc_repeatability_reverse(interest_points, interest_points_subsampled, grid_size * 1.5)
            print("Subsampled rep. value {}".format(repeatability_value_subsampled))
            subsampling_list.append(repeatability_value_subsampled)

        #Gaussian noise
        noise_levels = [0.1, 0.25, 0.5,  0.75, 1.0,1.25 ,1.5,1.75, 2.0,2.5 ]
        noise_list = []
        absolute_noise_factor = 0.001
        path_noise = os.path.join(path,"noise/")
        for noise_level in noise_levels:
            path_noise_tmp = path_noise + str(noise_level)
            interest_points_noise = self.gaus_noise_transformation(self.mesh_objects, noise_level, absolute_noise_factor,write_ply = write_ply, path = path_noise_tmp)
            repeatability_value_noise = self.calc_repeatability_reverse(interest_points, interest_points_noise, self.repeatbiliy_thresh)
            noise_list.append(repeatability_value_noise)
            print("Subsampled nosie. value {}".format(repeatability_value_noise))

        #Local holes, Micro Hole
       
        hole_size = 0.001
        combinations = [(5,hole_size),(10,hole_size),(20,hole_size),(50,hole_size),(75,hole_size),(100,hole_size)]
        micro_holes_list = []
        path_micro_holes = os.path.join(path,"micro_holes/")
        for combination in combinations:
            nb_holes, hole_size = combination
            path_micro_holes_tmp = path_micro_holes + str(nb_holes)
            interest_points_holes = self.local_holes_transformation( self.mesh_objects, nb_holes = nb_holes, hole_size = hole_size, write_ply = write_ply, path = path_micro_holes_tmp)
            repeatability_value_holes = self.calc_repeatability_reverse(interest_points, interest_points_holes, self.repeatbiliy_thresh)
            print("Holes {}".format(repeatability_value_holes))
            micro_holes_list.append(repeatability_value_holes)
        
       


        #Local holes, Holes
        
        combinations = [(20,0.0001),(20,0.0005),(20,0.001),(20,0.01)]
        holes_list = []
        path_holes = os.path.join(path,"holes/")
        for combination in combinations:
            nb_holes, hole_size = combination
            path_holes_tmp = path_holes + str(hole_size)
            interest_points_holes = self.local_holes_transformation( self.mesh_objects, nb_holes = nb_holes, hole_size = hole_size, write_ply= write_ply, path = path_holes_tmp)
            repeatability_value_holes = self.calc_repeatability_reverse(interest_points, interest_points_holes, self.repeatbiliy_thresh)
            print("Holes {}".format(repeatability_value_holes))
            holes_list.append(repeatability_value_holes)
        
        result_list = {"angle":angle_repeat_list,
                        "scale": scale_list,
                        "transl": repeatability_value_transl, 
                        "resolution":subsampling_list,
                        "noise_list": noise_list,
                        "micro_holes_list":micro_holes_list,
                        "holes_list":holes_list}
        return result_list


class Experiment_move(Experiment): 
    def __init__(self,path_file, sel_mod = "rel", self_args = {'thresh': 0.01},\
         neigh_args = {'k':10}, neigh_flag = "k", k_harris = 0.04): 
        
        pos, G = self.move_data(path_file)
        self.mesh_objects =base.Meshgrid(pos, G)
        self.repeatbiliy_thresh = self.mesh_objects.diameter * 0.01
        self.sel_mod = sel_mod
        self.sel_args = self_args
        self.neigh_args = neigh_args
        self.neigh_flag = neigh_flag 
        self.k_harris = k_harris 
        
    def move_data(self, datapath):
        with open(datapath + ".tri") as f:
               lines = f.read().splitlines()
        lines = [int(number) for line in lines for number in line.split(" ")]
        tri = np.array(lines).reshape(-1,3)-1 # 1 such that the index is equal the number

        with open(datapath + ".vert") as f:
            lines = f.read().splitlines()
        lines = [float(number) for line in lines for number in line.split(" ")]
        vert = np.array(lines).reshape(-1,3)
        G = nx.Graph()
        for i in range(tri.shape[0]): 
            G.add_edge(tri[i,0],tri[i,1])
            G.add_edge(tri[i,0],tri[i,2])
            G.add_edge(tri[i,1],tri[i,2])
        return vert, G


    def test_pipeline_move(self, path_file_tmps = [],write_ply = False):
        #Original interest points
        interest_points, _ =  self.mesh_objects.get_interest_points(sel_mod =self.sel_mod,\
            sel_args = self.sel_args,neigh_flag = self.neigh_flag, neigh_args = self.neigh_args,\
                 k_harris = self.k_harris, index = True)
        
        basename = path_file_tmps[0].split("/")[-1]
        basename = os.path.join("visualisation", basename) 
        if write_ply: 
            ply.write_ply(basename, interest_points, ['x', 'y', 'z'])
        
        move_list = []


        for path_file_tmp in path_file_tmps:  
            print("Path file tmp {}".format(path_file_tmp))
            pos_tmp, G_tmp = self.move_data(path_file_tmp)
            mesh_object_tmp =base.Meshgrid(pos_tmp, G_tmp)
            assert not np.all(np.isnan(pos_tmp)) and not np.all(np.isinf(pos_tmp))
            ip_moved, interest_points_index =  mesh_object_tmp.get_interest_points(sel_mod =self.sel_mod,\
                sel_args = self.sel_args,neigh_flag = self.neigh_flag, neigh_args = self.neigh_args,\
                     k_harris = self.k_harris, index = True)
            
            basename = path_file_tmp.split("/")[-1]
            basename = os.path.join("visualisation", basename) 
            if write_ply: 
                ply.write_ply(basename, ip_moved, ['x', 'y', 'z'])
            
            interest_points_moved = self.mesh_objects.np_position_array[interest_points_index]
            rep = self.calc_repeatability_reverse(interest_points, interest_points_moved,self.repeatbiliy_thresh)
            print("Rep {}".format(rep))
            move_list.append(rep)
        return move_list
if __name__ == "__main__":
    
    print("Results K-Neighbors")
    exp = Experiment("bunny.ply",sel_mod = "rel", self_args = {'thresh': 0.01},\
         neigh_args = {'k':10}, neigh_flag = "k", k_harris = 0.04)
    exp.test_pipeline()
    print("Results Distance")
    exp2 = Experiment("bunny.ply",sel_mod = "rel", self_args = {'thresh': 0.01},\
         neigh_args = {'distance':0.01}, neigh_flag = "dist", k_harris = 0.04)
    exp2.test_pipeline()

    print("Adaptive Ring")
    exp3 = Experiment("bunny.ply",sel_mod = "rel", self_args = {'thresh': 0.01},\
         neigh_args = {'max_dist':0.01}, neigh_flag = "ring", k_harris = 0.04)
    exp3.test_pipeline()
    #exp3.test_pipeline()