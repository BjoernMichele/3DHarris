import networkx

class Meshgrid:

    def __init__(np_position_array): 
        #Param: np_position_array: 

    
    def get_ring_neighbors(node_idx, k):
    '''
    returns 
    list of coordinates and 
    index of questioned node in that list.
    '''
    
    def get_k_nearest_neighbors
    
    def get_distance_neighbors
    
    def compute_all_harris_responses():
        '''
        Loop of point_cloud_2_harris_response for all points
        
        Fills self.harris_values
        '''

def normalize_point_cloud(points, idx_v):
    '''
    Centration on centroid,
    PCA,
    Rotation,
    Centration on v
    
    Returns transformed points
    '''
    
    
def fit_quadratic_surface(points):
    '''
    Returns parameters p1,...,p6
    '''
    

def point_cloud_2_harris_response(neighborhood, idx_v):
    '''
    calls neighborhood normalization
    calls surface fitting
    computes matrix
    
    returns Harris response
    '''

