import networkx as nx
from scipy.spatial import Delaunay
import numpy as np

#Taken from https://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-in-a-delaunay-triangulation-using-sci?rq=1
def find_neighbors(pindex, triang):
    return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]


class Meshgrid:

    def __init__(self,np_position_array): 
        #Param: np_position_array:
        self.np_position_array = np_position_array 
        self.G = self.create_graph(np_position_array)
    
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

    def get_ring_neighbors(node_idx, k):
        '''
        returns 
        list of coordinates and 
        index of questioned node in that list.
        '''
        pass
    
    def get_k_nearest_neighbors():
        pass
    
    def get_distance_neighbors():
        pass
    def compute_all_harris_responses():
        '''
        Loop of point_cloud_2_harris_response for all points
        
        Fills self.harris_values
        '''
        pass
def normalize_point_cloud(points, idx_v):
    '''
    Centration on centroid,
    PCA,
    Rotation,
    Centration on v
    
    Returns transformed points
    '''
    pass
    
    
def fit_quadratic_surface(points):
    '''
    Returns parameters p1,...,p6
    '''
    pass
    

def point_cloud_2_harris_response(neighborhood, idx_v):
    '''
    calls neighborhood normalization
    calls surface fitting
    computes matrix
    
    returns Harris response
    '''
    pass

