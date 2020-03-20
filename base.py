import networkx as nx
import numpy as np
import scipy
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation


class Meshgrid:

    def __init__(self, np_position_array, graph): 
        self.points = np_position_array
        self.graph = graph
        self.kdtree = None
        self.harris_values = None

    
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
    
    
    def compute_all_harris_responses(self, neighborhood_method):
        '''
        Loop of point_cloud_2_harris_response for all points
        
        Fills self.harris_values
        '''
        num_points = self.points.shape[0]
        self.harris_values = np.zeros(num_points)
        for idx in range(num_points):
            # TODO: How to choose and call neighborhood method?
            neighborhood, new_idx = neighborhood_method(idx)
            harris = point_cloud_2_harris_response(neighborhood, new_idx)
            self.harris_values[idx] = harris
            
        return harris

    
def compute_normal(points):
    '''
    Points shall be a Nx3 array (N 3D-points)
    Performs a PCA and returns the eigenvector corresponding to the smallest singularvalue
    '''
    # Center data
    centroid = np.mean(points, axis=0)
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


def normalize_point_cloud(points, idx_v):
    '''
    Centration on centroid,
    PCA,
    Rotation (normal onto z-axis),
    Centration on v
    
    Returns transformed points
    '''
    
    # center on centroid
    centroid = np.mean(points, axis=0)
    new_points = points
    new_points -= centroid
    
    # rotate
    normal = compute_normal(new_points)
    R = get_rotation(normal, np.array([0,0,1]))
    new_points = R.apply(new_points)
    
    # center on v
    new_points -= new_points[idx_v]
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

    

def point_cloud_2_harris_response(neighborhood, idx_v, k):
    '''
    calls neighborhood normalization
    calls surface fitting
    computes matrix
    
    returns Harris response
    '''
    normalized_neighborhood = normalize_point_cloud(neighborhood, idx_v)
    p = fit_quadratic_surface(normalized_neighborhood)
    
    # See paper, formulas (10)-(12)
    A = p[3]**2 + 2*p[0]**2 + 2*p[1]**2
    B = p[4]**2 + 2*p[1]**2 + 2*p[2]**2
    C = p[3]*p[4] + 2*p[0]*p[1] + 2*p[1]*p[2]
    E = np.array([[A,C],[C,B]])
    harris = (A*B-C**2) - k*((A+B)**2) # det(E) - k*(trace(E)**2)
    return harris
    

