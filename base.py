import networkx as nx
from scipy.spatial import Delaunay
import scipy
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation
import numpy as np
from numba import jit
import numba




def find_neighbors(pindex, triang):
    #Taken from https://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-in-a-delaunay-triangulation-using-sci?rq=1
    return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]


#This class provides the basic functions for the Harris response value calculation, different local neighborhood selections, local neighborhood normalization and quadratic surface
# approximation. 
class Meshgrid:
    
    def __init__(self,np_position_array, graph = None, diameter_set=None):
        #Param: np_position_array:
        self.points = np_position_array
        self.np_position_array = np_position_array
        if graph is None:  
            self.graph = self.create_graph(np_position_array)
        else: 
            self.graph = graph
        self.kdtree = None
        self.harris_values = None
        self.calculate_size(self.points, diameter_set)

    def calculate_size(self, np_position_array, diameter_set): 
        self.width = np.max(np_position_array[:,0]) - np.min(np_position_array[:,0])
        self.height = np.max(np_position_array[:,1]) - np.min(np_position_array[:,1])
        self.depth = np.max(np_position_array[:,2]) - np.min(np_position_array[:,2])
        if diameter_set is None: 
            self.diameter = np.sqrt(self.width**2+ self.height**2 + self.depth**2)
        else: 
            self.diameter = diameter_set

    def create_graph(self, np_position_array):
        #Create the Delauny Triangulation
        tri = Delaunay(np_position_array)
        #Test if there points which are not in the planar part
        assert tri.coplanar.shape[0] == 0 #Checks that all points are used

        #Create the Graph with the connections
        G = nx.Graph()
        G.add_nodes_from(range(0, np_position_array.shape[0]))
        for i in range(len(np_position_array)):
            #All nodes of one simplex are considered as neighbors in the graph
            neighbors_i = find_neighbors(i, tri)
            [G.add_edge(i, neigh) for neigh in neighbors_i]

        return G


    
    def get_ring_neighbors(self, node_idx, max_dist):
        '''
        Returns a list of coordinates contained in the ring_neighborhood
        and the index of the questioned node in that list
        '''
        current_ring = set([node_idx])
        all_seen_nodes = set([node_idx])
        go_to_next_ring = True
        while go_to_next_ring:
            next_ring = set()
            # Get next_ring nodes
            for ring_node in current_ring:
                neighs = self.graph.adj[ring_node]
                for n in neighs:
                    if n not in all_seen_nodes:
                        next_ring.add(n)

            # Check that they are close enough
            for n in next_ring:
                dist = np.linalg.norm(self.points[node_idx] - self.points[n])
                if dist > max_dist:
                    go_to_next_ring = False
                    break

            if go_to_next_ring:
                all_seen_nodes.update(next_ring)
                current_ring = next_ring

        # Return found rings
        neighbor_indices = np.array([node_idx]+list(all_seen_nodes - set([node_idx])))
        idx = 0
        return self.points[neighbor_indices], idx

   
    def get_k_nearest_neighbors(self, node_idx, k):
        '''
        Returns a list of coordinates of the k nearest neighbors and the node itself (k+1 points)
        and the index of the node itself in that list
        '''
        if self.kdtree is None:
            self.kdtree = KDTree(self.points)
        indices = self.kdtree.query(self.points[node_idx].reshape(1,-1), k=k+1, return_distance=False, sort_results=False)[0]
        # points[node_idx] is also contained in its own neighborhood
        idx = np.argwhere(indices==node_idx)[0][0]
        return self.points[indices], idx

    
    def get_distance_neighbors(self, node_idx, distance):
        '''
        Returns a list of coordinates of a spatial neighborhood
        and the index of the questioned node in that list
        '''
        if self.kdtree is None:
            self.kdtree = KDTree(self.points)
        indices = self.kdtree.query_radius(self.points[node_idx].reshape(1,-1), r=distance )[0]
        # points[node_idx] is also contained in its own neighborhood
        idx = np.argwhere(indices==node_idx)[0][0]
        return self.points[indices], idx

    def get_interest_points(self,sel_mod ="rel", sel_args = {'thresh': 0.01},neigh_flag = 'ring', neigh_args = None, k_harris = 0.04, index=False ):
        self.compute_all_harris_responses( neigh_flag = neigh_flag, neigh_args= neigh_args,  k_harris = k_harris )
        interest_points = None
        if sel_mod == 'rel': 
            interest_points, ip_idx = self.interest_selection_relative(threshold_factor = sel_args['thresh'])
        elif sel_mod == 'distr': 
            interest_points = self.interest_selection_distributed(threshold_factor = sel_args['thresh'])
        else: 
            raise ValueError("Unknown interest points selection method")
        if index: 
            #Returns interest points and index
            return interest_points, ip_idx
        else:
            #Returns only interest points 
            return interest_points
    
    def interest_selection_relative(self,threshold_factor = 0.01): 
        #Returns all points that 
        assert threshold_factor <= 1.0
        assert self.harris_values.shape[0] > 0
        idx_sorted = np.argsort(self.harris_values)[::-1] #Gives back the indices for descending order
        number_sampled_points = max(1,int(threshold_factor * self.harris_values.shape[0]))
        interest_points_idx = idx_sorted[:number_sampled_points]
        assert interest_points_idx.shape[0] > 0
        interest_points = self.points[interest_points_idx]
        self.interest_points_relative = interest_points
        return interest_points, interest_points_idx
    def interest_selection_distributed(self, threshold_factor):
        #Returns the interested points, selected based on the distance to already
        # sampled interest points (threshold factor gives the relation to the diameter of the object)
        threshold = self.diameter * threshold_factor
        idx_sorted = np.argsort(self.harris_values)[::-1] #Gives back the indices for descending order
        interest_points = np.expand_dims(self.points[idx_sorted[0]],0)
        for idx in idx_sorted[1:]: 
                new_position = np.expand_dims(self.points[idx], 0) 

                differences = np.any(np.linalg.norm(interest_points - new_position, axis = 1) < threshold)
                if not differences: 
                    interest_points = np.vstack((interest_points, new_position))
        self.interest_points_distributed = interest_points
        return interest_points
    
    
    def compute_all_harris_responses(self, neigh_flag = 'ring', neigh_args = None,  k_harris = 0.04 ):

        '''
        Loop of point_cloud_2_harris_response for all points

        Fills self.harris_values
        '''

        num_points = self.points.shape[0]
        self.harris_values = np.zeros(num_points)
        for idx in range(num_points):
            if neigh_flag == 'ring': 
                max_dist = neigh_args['max_dist'] * self.diameter
                #print("Max dist {}".format(max_dist))
                neighborhood, new_idx = self.get_ring_neighbors(idx, max_dist)
            elif neigh_flag == 'k': 
                k = neigh_args['k']
                neighborhood, new_idx = self.get_k_nearest_neighbors(idx, k)
            elif neigh_flag == 'dist':
                distance = neigh_args['distance'] * self.diameter
                #print("L2 distance {}".format(distance)) 
                neighborhood, new_idx = self.get_distance_neighbors(idx, distance)
            else: 
                raise ValueError("Unknown neighborhood method")
            
            assert not np.all(np.isnan(neighborhood)) and not np.all(np.isinf(neighborhood))
            if neighborhood.shape[0] == 0: 
                print(neighborhood)
            #print(neighborhood)
            harris = point_cloud_2_harris_response(neighborhood, new_idx,k_harris)
            self.harris_values[idx] = harris

        return harris

    def compute_all_harris_responses_debug(self, idx ,neigh_flag = 'ring', neigh_args = None,  k_harris = 0.04 ):

        '''
        Loop of point_cloud_2_harris_response for all points

        Debug functions, and therefore returns intermediate results
        '''

        num_points = self.points.shape[0]
        self.harris_values = np.zeros(num_points)
        
        if neigh_flag == 'ring': 
            max_dist = neigh_args['max_dist'] * self.diameter
            neighborhood, new_idx = self.get_ring_neighbors(idx, max_dist)
        elif neigh_flag == 'k': 
            k = neigh_args['k']
            neighborhood, new_idx = self.get_k_nearest_neighbors(idx, k)
        elif neigh_flag == 'dist':
            distance = neigh_args['distance'] * self.diameter 
            neighborhood, new_idx = self.get_distance_neighbors(idx, distance)
        else: 
            raise ValueError("Unknown neighborhood method")
        
        assert not np.all(np.isnan(neighborhood)) and not np.all(np.isinf(neighborhood))
        if neighborhood.shape[0] == 0: 
            print(neighborhood)
        #print(neighborhood)
        orig_neighborhood = np.copy(neighborhood)
        centroid, normal, harris,normalized_neighborhood,new_points_before_centered, p = point_cloud_2_harris_response(neighborhood, new_idx,k_harris, debug = True)
        self.harris_values[idx] = harris

        return harris,neighborhood,orig_neighborhood, centroid, normal,normalized_neighborhood,new_points_before_centered,p,new_idx, self.graph

def compute_normal(points):
    '''
    Points shall be a Nx3 array (N 3D-points)
    Performs a PCA and returns the eigenvector corresponding to the smallest singularvalue
    '''
    # Center data
    centroid = np.mean(points, 0)
    
    points_centered = points - centroid[None,:]
    # Compute covariance matrix
    cov = 1/points.shape[0] * points_centered.T @ points_centered
    # Compute eigenvalues
    eigvalues, eigvectors = np.linalg.eigh(cov)
    return eigvectors.T[0]


def get_rotation(src, tgt):
    '''
    Returns a rotation r that rotates src onto tgt
    '''
    n_src = src/np.linalg.norm(src)
    n_tgt = tgt/np.linalg.norm(tgt)
    v = np.cross(n_src,n_tgt)
    # v is the normed rotation axis
    v /= np.linalg.norm(v)
    # rotation angle alpha
    alpha = np.arccos(np.sum(n_src*n_tgt))
    return Rotation.from_rotvec(v*alpha)


def normalize_point_cloud(points, idx_v, debug = False):
    '''
    Centration on centroid,
    PCA,
    Rotation (normal onto z-axis),
    Centration on v

    Returns transformed points
    '''

    # center on centroid
    centroid = np.mean(points, 0)
    new_points = points
    new_points -= centroid

    # rotate
    normal = compute_normal(new_points)
    R = get_rotation(normal, np.array([0,0,1]))
    new_points = R.apply(new_points)

    # center on v
    new_points_before_centered = np.copy(new_points)
    new_points -= new_points[idx_v]
    if debug: 
        return centroid, normal, new_points, new_points_before_centered
    return new_points

def fit_quadratic_surface(points):
    '''
    Returns parameters p1,...,p6
    '''
    # Creates a matrix with x**2, xy, y**2, ...
    # Then fits an mse model
    matrix = np.ones((points.shape[0],6))
    #             p1     p2   p3    p4    p5    p6
    exponents = [(2,0),(1,1),(0,2),(1,0),(0,1),(0,0)]
    for k, (i, j) in enumerate(exponents):
        matrix[:, k] = points[:,0]**i * points[:,1]**j
    m, _, _, _ = scipy.linalg.lstsq(matrix, points[:,2])
    return m


def point_cloud_2_harris_response(neighborhood, idx_v, k, debug = False):
    '''
    calls neighborhood normalization
    calls surface fitting
    computes matrix

    returns Harris response
    '''
    if debug: 
        centroid, normal, normalized_neighborhood,new_points_before_centered= normalize_point_cloud(neighborhood, idx_v, debug = True)
        
    else: 
        normalized_neighborhood = normalize_point_cloud(neighborhood, idx_v)
    p = fit_quadratic_surface(normalized_neighborhood)

    # See paper, formulas (10)-(12)
    A = p[3]**2 + 2*p[0]**2 + 2*p[1]**2
    B = p[4]**2 + 2*p[1]**2 + 2*p[2]**2
    C = p[3]*p[4] + 2*p[0]*p[1] + 2*p[1]*p[2]
    E = np.array([[A,C],[C,B]])
    harris = (A*B-C**2) - k*((A+B)**2) # det(E) - k*(trace(E)**2)
    
    if debug:
        return centroid, normal, harris,normalized_neighborhood,new_points_before_centered, p
    else:
        return harris
