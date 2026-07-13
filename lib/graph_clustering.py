# Graph benchmark and clustering helpers.
#
# Imported and adapted from the FPGW (Fused Partial Gromov-Wasserstein) repository:
#   https://github.com/yikun-baio/fused-pgw
# The algorithms in this file originate there; SpaOT is a downstream user.
# Some functions have been modified for SpaOT analyses.

import sys

import numpy as np
import ot
import torch
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.utils import check_random_state

from .fgw.fgw import fgw_barycenters, fgw_lp
from .fgw.fgwclustering import EmptyClusterError, _check_no_empty_cluster
from .fgw.graph import Graph
from .fugw.utils import _make_tensor, console
from .fused_pgw import fused_partial_gromov_wasserstein, fused_pgw_cost
from .fused_pgw_barycenter import fmpgw_barycenters, fugw_barycenters

def build_comunity_graph(N=30,Nc=3,sigma=0.3,pw=0.8,pb=0.2,seed=None):
    """Generate a synthetic community graph with scalar node attributes."""
    if seed is not None:
        np.random.seed(seed)
    c=(Nc*np.arange(N)/N).astype(int)
    c2=(2*Nc*np.arange(N)/N).astype(int)
    v=c+1*np.mod(c2,2)+sigma*np.random.randn(N)
    g=Graph()
    g.add_nodes(list(range(N)))
    for i in range(N):
         g.add_one_attribute(i,v[i])
         for j in range(i+1,N):
             r=np.random.rand()
             if (c[i]==c[j]) or ((c[i]==c[j]-1) and r<pb): # or (c[i]==0 and c[j]==Nc)
                 g.add_edge((i,j))
         
    return g,v

def build_dataset(M=100,all_N=[30,40],type_list=[1,2,3,4],seed1=None,seed2=None):
    # X the list of community graphs
    # V the features
    # y the class
    """Generate labeled synthetic graph datasets for clustering benchmarks."""
    X=[]
    y=[]
    V=[]
    if seed1 is not None:
        np.random.seed(seed1)
    for i in range(M):
        for nc in type_list:
            n=np.random.choice(all_N)
            g,v=build_comunity_graph(N=n,Nc=nc,sigma=0.1,pw=0.7,pb=0.1,seed=seed2)
            X.append(g)
            V.append(v)
            y.append(nc)
    return X,y,V
    
#%% Heuristic to compute a adjency matrix from a C in R^n\times n
def sp_to_adjency(C,threshinf=0.2,threshsup=1.8):
    """Threshold a structural distance matrix into an adjacency matrix."""
    H=np.zeros_like(C)
    np.fill_diagonal(H,np.diagonal(C))
    C=C-H
    #C=stats.threshold(C, threshmin=threshinf, threshmax=threshsup, newval=0)
    C=np.minimum(np.maximum(C,threshinf),threshsup)
    C[C==threshsup]=0
    C[C!=0]=1   
    
    return C   
#%%
from scipy.sparse.csgraph import shortest_path
def find_thresh(C,inf=0.5,sup=3,step=10):
    """Search for an adjacency threshold whose shortest-path matrix matches a target structure."""
    dist=[]
    search=np.linspace(inf,sup,step)
    for thresh in search:
        Cprime=sp_to_adjency(C,0,thresh)
        #print(Cprime)
        SC=shortest_path(Cprime,method='D')
        SC[SC==float('inf')]=100
        #print(SC)
        dist.append(np.linalg.norm(SC-C))
    return search[np.argmin(dist)],dist




def add_outliers(x,eta=0.2,outlier_id_min=50,attr_range=2,edge_type=0,gap=0):
    # determine number of outliers
    """Add synthetic outlier nodes and edges to a graph benchmark instance."""
    num_outlier=int(eta*x.N)
    if num_outlier==0:
        num_outlier+=1
        
    # obtain the index list and largest index for all nodes 
    node_id_list=list(x.nodes().keys())
    node_id_list=np.array(node_id_list, dtype=int)
    # outlier_id_min=max(node_id_list)+10
    
    # obtain the attribution and the range of attributes of all nodes
    node_attri_list=np.array(x.values())
    node_attri_max=max(node_attri_list)
    node_attri_max_upper_bound=node_attri_max+3*node_attri_list.std(0)
    
    # select node
    outlier_id_list=np.arange(outlier_id_min,outlier_id_min+num_outlier)
    
    x.add_nodes(outlier_id_list)
    attr=node_attri_max +gap+ np.random.rand(num_outlier) * (attr_range*node_attri_list.std())
    for i, node_id in enumerate(outlier_id_list):
        x.add_one_attribute(node_id,attr[i],attr_name='attr_name')
        
    edges_list=defines_outlier_edge(outlier_id_list,node_id_list,seed=0,edge_type=edge_type)
    for edge in edges_list:
        x.add_edge(edge)
    return None



def cdist_fpgw(X_features, X_structure, Y_features, Y_structure, alpha,metric='sqeuclidean',measure='fgw',N_list=None):

    """Compute pairwise SpaOT/FPGW distances between two graph collections."""
    n_X = len(X_features)
    n_Y = len(Y_features)

    # TMP #
    for i in range(n_X):
        assert np.linalg.norm(X_structure[i] - X_structure[i].T) < 1e-5, "Wooops X"
    for j in range(n_Y):
        assert np.linalg.norm(Y_structure[j] - Y_structure[j].T) < 1e-5, "Wooops Y"


    dists = np.empty((n_X, n_Y))
    for i in range(n_X):
        for j in range(n_Y):
            dist_features_ij = cdist(np.array(X_features[i]).reshape((len(X_features[i]), -1)),
                                     np.array(Y_features[j]).reshape((len(Y_features[j]), -1)),
                                     metric=metric)
            C1,C2=X_structure[i],Y_structure[j]
            if measure=='fgw':
                p,q=np.ones(len(X_features[i]), ) / len(X_features[i]),np.ones(len(Y_features[j]), ) / len(Y_features[j])
                transport, log = fgw_lp((1. - alpha)*dist_features_ij,C1,C2,p,q,loss_fun='square_loss',
                                        alpha=alpha,verbose=False,log=True)
                # TODO: test wgw ? (reg) regarder GW_dist
                dists[i, j] = log["loss"][-1]
   
            if measure=='fmpgw':
                p=np.ones(len(X_features[i]))/N_list[i]
                q=np.ones(len(Y_features[j]))/len(Y_features[j]) # Transport all mass of Y (centers)
                #mass=min(p.sum(),q.sum())
                transport= fused_partial_gromov_wasserstein(dist_features_ij,C1,C2,p,q,omega2=alpha,mass=None,
                                                                  loss_fun='square_loss',verbose=False,log=False)
                cost=fused_pgw_cost(dist_features_ij,C1,C2,transport,alpha,loss_fun='square_loss')
                dists[i,j]=cost
                
            if measure=='fugw':
                p=np.ones(len(X_features[i]))/N_list[i]
                q=np.ones(len(Y_features[j]))/len(Y_features[j]) # Transport all mass of Y (centers)
                #mass=min(p.sum(),q.sum())
                rho,eps=1.0,0.05

                gamma,_,log=ot.gromov.fused_unbalanced_gromov_wasserstein(Cx=C1, Cy=C2,wx=p,wy=q,reg_marginals=rho, epsilon=eps,
                                                                divergence="kl",unbalanced_solver="mm",alpha=alpha,M=dist_features_ij,log=True)
                cost=log['fugw_cost']
                dists[i,j]=cost    
            
    return dists

#@nb.njit(cache=True)
def ensure_2d(x):
    """Convert a one-dimensional feature array to a two-dimensional matrix."""
    if x.ndim == 1:
        x = x[:, np.newaxis]  # Convert (n,) to (n, 1)
    return x
    
def X_to_features(X):
    """Extract feature and structure matrices from a list of Graph objects."""
    X_features=[np.asarray(x.values()) for x in X]
    X_features2=[ensure_2d(x) for x in X_features]
    return X_features2


            



def defines_outlier_edge(node_outlier_id_list,nodes_id_list,seed=0,edge_type=0):
    """Construct edges connecting synthetic outlier nodes to a graph."""
    outlier_edge_list=[]
    n=len(node_outlier_id_list)
    np.random.seed(seed)
    if edge_type==2:
        for i in range(n-1):
            edge=(node_outlier_id_list[i],node_outlier_id_list[i+1])
            outlier_edge_list.append(edge)
        edge=(nodes_id_list[0],node_outlier_id_list[0])
        outlier_edge_list.append(edge)
    
    if edge_type==1:
        selected_elements = np.random.choice(nodes_id_list, size=n, replace=False)
        for i in range(n):        
            edge=(selected_elements[i],node_outlier_id_list[i])
            outlier_edge_list.append(edge)
    if edge_type==0:
        # Union of A and B
        A_union_B = np.concatenate((nodes_id_list, node_outlier_id_list))
        
        # Result array to store the chosen elements
        #result = np.empty_like(node_outlier_id_list)

        # For each element in A, choose an element from A_union_B that is not the same as the current element in A
        a=node_outlier_id_list[0]
        b=np.random.choice(nodes_id_list)
        outlier_edge_list.append((b,a))
        for i, a in enumerate(node_outlier_id_list[1:]):
            # Exclude the current element 'a' from the choices
            choices = A_union_B[A_union_B != a]
            # Randomly select one element from the remaining choices
            b = np.random.choice(choices)
            edge= (b,a)
            outlier_edge_list.append(edge)
    return outlier_edge_list

class FusedGromovWassersteinGraphKMeans1():
    """K-means clustering with FGW for graph data.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.
    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm stops.
    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.
    max_iter_barycenter : int (default: 100)
        Number of iterations for the barycenter computation process.
    metric_params : dict or None
        Parameter values for the chosen metric.
        Value associated to the `"alpha"` key corresponds to the alpha parameter in Fused-Gromov-Wasserstein.
        Default value is 0.5.
        Value associated to the `"centroid_sz"` key defines the size (in number of timestamps) of the obtained centroids.
        Default value is `None` which means the size of the first time series in the dataset will be used.
        Value associated to the `"fixed_structure"` key defines whether to use a fixed (regular) structure for the.
        barycenters or not. Default value is `False`.
        Value associated to the `"line_search_method"` key defines the method to be used for line search during
        optimization. Default value is `"amijo"`.
    verbose : {0, 1, 2} (default: 1)
        Verbose level: 0 means no message, 1 means messages only from the clustering part, 2 means messages from both
        clustering and FGW

    Attributes
    ----------
    labels_ : np.ndarray
        Labels of each point.
    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """

    def __init__(self,N=None, Nc=3, n_clusters=3, max_iter=50, tol=1e-4, n_init=1, max_iter_barycenter=100,
                 metric_params=None, verbose=1, random_state=None,max_attempts=10,b=0.9,a=0.5,measure='fgw'):

        if metric_params is None:
            metric_params = {}
        self.alpha_fgw = metric_params.get("alpha", 0.5)
        self.fixed_structure = metric_params.get("fixed_structure", False)
        self.fixed_feature=metric_params.get("fixed_feature", False)
        self.bar_structure=metric_params.get("bar_structure", None)
        self.bar_feature=metric_params.get("bar_feature", None)
        self.line_search_method = metric_params.get("line_search_method", "amijo")
        self.max_attempts=max_attempts
        self.max_iter=max_iter

        self.n_clusters=n_clusters

        self.max_iter_barycenter=max_iter_barycenter
        self.tol=tol
        self.n_init=n_init
        self.verbose=verbose
        self.random_state=random_state

        self._cluster_centers_features = None
        self._cluster_centers_structure = None
        self.labels_=None
        self.inertia_ = np.inf
        self.all_cluster_centers_features={}
        self.all_cluster_centers_structure={}
        self.N=N
        self.measure=measure
        self.a=a
        self.b=b
        self.Nc=Nc

    def compute_inertia(self,distances,assignments):
        n=distances.shape[0]
        return np.sum(distances[np.arange(n), assignments]) / n

    def _assign_fgw(self, X, structural_information, update_class_attributes=True):

        #print('_assign_fgw')
        N_list=[x.N for x in X]
        dists = cdist_fpgw(X_features=[x.values() for x in X],
                          X_structure=structural_information,
                          Y_features=self._cluster_centers_features,
                          Y_structure=self._cluster_centers_structure,
                          alpha=self.alpha_fgw,
                          N_list=N_list,
                          measure=self.measure)
        matched_labels = dists.argmin(axis=1)

        if update_class_attributes:
            #print("update_class_attributes")
            self.labels_ = matched_labels
            _check_no_empty_cluster(self.labels_, self.n_clusters)
            inertia_dists = dists
            self.inertia_ = self.compute_inertia(inertia_dists, self.labels_)
        return matched_labels

    def _update_centroids(self,X,it):

        for k in range(self.n_clusters):
            cluster_data = [x.values() for x in X[self.labels_ == k]]
            structural_information = [x.C for x in X[self.labels_ == k]]
            if self.N is None:
                centroid_mean_size=int(np.mean([len(x.nodes()) for x in X[self.labels_ == k]])) #size of the centroid in the cluster is the mean size of the nodes of the graphs in the cluster
            else:    
                centroid_mean_size=self.N
            num_x=len(cluster_data)
                        #np.ones(num)/num
            
            graph_weights = np.ones(num_x)/num_x
            #np.ones(len(internal_weights)) / len(internal_weights)
            if self.measure=='fgw':
                internal_weights =[np.ones(len(x.nodes())) / len(x.nodes()) for x in X[self.labels_ == k]] 
                features, structure, _ = fgw_barycenters(N=centroid_mean_size,
                                                         Ys=cluster_data,
                                                         Cs=structural_information,
                                                         ps=internal_weights,
                                                         lambdas=graph_weights,
                                                         alpha=self.alpha_fgw,
                                                         max_iter=self.max_iter_barycenter,
                                                         fixed_structure=False,
                                                         fixed_features=False,
                                                         init_X=self._cluster_centers_features[k],
                                                         init_C=self._cluster_centers_structure[k],
                                                         verbose=(self.verbose == 2))
            elif self.measure=='fmpgw':
                internal_weights = [np.ones(len(x.nodes())) / x.N for x in X[self.labels_ == k]]
                mass_list=[min(1,p.sum()) for p in internal_weights]
                features, structure = fmpgw_barycenters(N=centroid_mean_size,
                                                         Ys=cluster_data,
                                                         Cs=structural_information,
                                                         ps=internal_weights,
                                                         lambdas=graph_weights,
                                                         alpha=self.alpha_fgw,
                                                         mass_list=mass_list,
                                                         max_iter=self.max_iter_barycenter,
                                                         fixed_structure=False,
                                                         fixed_features=False,
                                                         init_X=self._cluster_centers_features[k],
                                                         init_C=self._cluster_centers_structure[k],
                                                         verbose=False)
            elif self.measure=='fugw':
                internal_weights = [np.ones(len(x.nodes())) / x.N for x in X[self.labels_ == k]]
                mass_list=[min(1,p.sum()) for p in internal_weights]
                features, structure = fugw_barycenters(N=centroid_mean_size,
                                                         Ys=cluster_data,
                                                         Cs=structural_information,
                                                         ps=internal_weights,
                                                         lambdas=graph_weights,
                                                         alpha=self.alpha_fgw,
                                                         mass_list=mass_list,
                                                         max_iter=self.max_iter_barycenter,
                                                         fixed_structure=False,
                                                         fixed_features=False,
                                                         init_X=self._cluster_centers_features[k],
                                                         init_C=self._cluster_centers_structure[k],
                                                         verbose=False)                
                
            self._cluster_centers_features[k] = features
            self._cluster_centers_structure[k] = structure

            self.all_cluster_centers_features[(it,k)]=features
            self.all_cluster_centers_structure[(it,k)]=structure

    def _fit_one_init_fgw(self, X, structural_information, rs,y=None):

        breaked=False
        
        #indices = rs.randint(low=0, high=len(X), size=self.n_clusters)
        #self._cluster_centers_features = [X[i].values() for i in indices]
        #self._cluster_centers_structure = [X[i].C for i in indices]

        self._cluster_centers_features=[]
        self._cluster_centers_structure=[]
        for k in range(self.n_clusters):
            if self.N is None:
                N=np.random.randint(15,20)
            else:
                N=self.N
            #Nc=np.random.randint(3,5)
            if self.Nc is None:
                Nc=3
            else:
                Nc=self.Nc
            g,v=build_comunity_graph(N,Nc,sigma=0.1,pw=0.7,pb=0.6)
            #self._cluster_centers_features.append((self.b-self.a)*np.random.rand(N)+self.a)
            #self._cluster_centers_structure.append(nx.adjacency_matrix(nx.fast_gnp_random_graph(N,1)).toarray())
            self._cluster_centers_features.append(v)
            C=g.distance_matrix(method='shortest_path',force_recompute=True)
            self._cluster_centers_structure.append(C)




        for k in range(self.n_clusters):
            self.all_cluster_centers_features[(0,k)]=self._cluster_centers_features[k]
            self.all_cluster_centers_structure[(0,k)]=self._cluster_centers_structure[k]

        old_inertia = np.inf

        for it in range(self.max_iter):
            self._assign_fgw(X, structural_information=structural_information)
            if self.verbose > 0:
                sys.stdout.write(" Inertia : %.3f  " % self.inertia_)
                if y is not None:
                    sys.stdout.write(" & MI : %.3f " % adjusted_mutual_info_score(y,self.labels_))
                sys.stdout.write(" -->  ")
                sys.stdout.flush()
            self._update_centroids(X,it+1)

            if np.abs(old_inertia - self.inertia_) < self.tol:
                self.attempted_iter=it+1
                breaked=True
                break
            old_inertia = self.inertia_
        if not breaked:
            self.attempted_iter=self.max_iter+1
        if self.verbose > 0:
            sys.stdout.write("\n")

        return self

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        """
        structural_information = [x.C for x in X]


        rs = check_random_state(self.random_state)

        best_correct_centroids = None
        min_inertia = np.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < self.max_attempts:
            try:
                if self.verbose > 0 and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init_fgw(X, structural_information, rs,y=y)

                if self.inertia_ < min_inertia:
                    best_correct_centroids = (self._cluster_centers_features.copy(),
                                              self._cluster_centers_structure.copy())
                    min_inertia = self.inertia_
                n_successful += 1
                n_successful += 1

            except EmptyClusterError:
                if self.verbose > 0:
                    print("Resumed because of empty cluster")
        self._post_fit_fgw(X, structural_information, best_correct_centroids, min_inertia)
        

        return self

    def predict(self, X_tuple):
        """Compute assignment for data in X_tuple.

        Parameters
        ----------
        X_tuple : pair of (feature matrix, list_of_structure_matrices)

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        X_, structural_information = X_tuple
        return self._assign_fgw(X_, structural_information=structural_information, update_class_attributes=False)

    def _post_fit_fgw(self, X_fitted, structural_information_fitted, centroids, inertia):
        if np.isfinite(inertia) and (centroids is not None):
            self._cluster_centers_features, self._cluster_centers_structure = centroids
            self._assign_fgw(X_fitted, structural_information_fitted)
            #self._compute_cluster_centers(X_fitted=X_fitted)
            self.X_fit_ = X_fitted
            self.inertia_ = inertia
        else:
            self.X_fit_ = None




class FUGWBarycenter:
    """FUGW barycenters"""

    def __init__(
        self,
        alpha=0.5,
        rho=1,
        eps=1e-4,
        reg_mode="joint",
        force_psd=False,
        learn_geometry=False,
    ):
        # Save model arguments
        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.reg_mode = reg_mode
        self.force_psd = force_psd
        self.learn_geometry = learn_geometry

    @staticmethod
    def update_barycenter_geometry(
        plans_, weights_, geometry_, force_psd, device
    ):
        barycenter_geometry = 0
        # pi_samp, pi_feat: both of size (ns, n)
        for i, (plans, weights) in enumerate(zip(plans_, weights_)):
            if len(geometry_) == 1 and len(weights_) > 1:
                C = _make_tensor(geometry_[0], device=device)
            else:
                C = _make_tensor(geometry_[i], device=device)

            pi_samp, pi_feat = plans
            pi1_samp, pi1_feat = pi_samp.sum(0), pi_feat.sum(0)

            if force_psd:
                if isinstance(C, tuple):
                    C1, C2 = C
                    term = pi_samp.T @ (C1 @ (C2.T @ pi_samp)) / (
                        pi1_samp[:, None] * pi1_samp[None, :]
                    ) + pi_feat.T @ (C1 @ (C2.T @ pi_feat)) / (
                        pi1_feat[:, None] * pi1_feat[None, :]
                    )
                elif torch.is_tensor(C):
                    term = pi_samp.T @ C @ pi_samp / (
                        pi1_samp[:, None] * pi1_samp[None, :]
                    ) + pi_feat.T @ C @ pi_feat / (
                        pi1_feat[:, None] * pi1_feat[None, :]
                    )
                term = term / 2

            else:
                if isinstance(C, tuple):
                    C1, C2 = C
                    term = (
                        pi_samp.T
                        @ (C1 @ (C2.T @ pi_feat))
                        / (pi1_samp[:, None] * pi1_feat[None, :])
                    )  # shape (n, n)
                elif torch.is_tensor(C):
                    term = (
                        pi_samp.T
                        @ C
                        @ pi_feat
                        / (pi1_samp[:, None] * pi1_feat[None, :])
                    )  # shape (n, n)

            w = _make_tensor(weights, device=device)
            barycenter_geometry = (
                barycenter_geometry + w * term
            )  # shape (n, n)

        return barycenter_geometry

    @staticmethod
    def update_barycenter_features(plans, features_list, device):
        for i, (pi, features) in enumerate(zip(plans, features_list)):
            # Use uniform weights across subjects
            weight = 1 / len(features_list)
            f = _make_tensor(features, device=device)
            if features is not None:
                acc = weight * pi.T @ f.T / (pi.sum(0).reshape(-1, 1) + 1e-16)

                if i == 0:
                    barycenter_features = acc
                else:
                    barycenter_features += acc

        return barycenter_features.T

    @staticmethod
    def get_dim(C):
        if isinstance(C, tuple):
            return C[0].shape[0]
        elif torch.is_tensor(C):
            return C.shape[0]

    @staticmethod
    def get_device_dtype(C):
        if isinstance(C, tuple):
            return C[0].device, C[0].dtype
        elif torch.is_tensor(C):
            return C.device, C.dtype

    def compute_all_ot_plans(
        self,
        plans,
        duals,
        weights_list,
        features_list,
        geometry_list,
        barycenter_weights,
        barycenter_features,
        barycenter_geometry,
        solver,
        solver_params,
        device,
        verbose,
    ):
        new_plans = []
        new_losses = []

        for i, (features, weights) in enumerate(
            zip(features_list, weights_list)
        ):
            if verbose:
                console.log(f"Updating mapping {i + 1} / {len(weights_list)}")

            if len(geometry_list) == 1 and len(weights_list) > 1:
                G = geometry_list[0]
            else:
                G = geometry_list[i]

            # mapping = FUGW(
            #     alpha=self.alpha,
            #     rho=self.rho,
            #     eps=self.eps,
            #     reg_mode=self.reg_mode,
            # )

            # mapping.fit(
            #     source_features=features,
            #     target_features=barycenter_features,
            #     source_geometry=G,
            #     target_geometry=barycenter_geometry,
            #     source_weights=weights,
            #     target_weights=barycenter_weights,
            #     init_plan=plans[i] if plans is not None else None,
            #     init_duals=duals[i] if duals is not None else None,
            #     solver=solver,
            #     solver_params=solver_params,
            #     device=device,
            #     verbose=verbose,
            # )
            M=ot.dist(features,barycenter_features)
            gamma,_,log=ot.gromov.fused_unbalanced_gromov_wasserstein(Cx=G, Cy=barycenter_geometry,wx=weights,wy=barycenter_weights,reg_marginals=rho,
                                                                      G0=plans[i],epsilon=eps,divergence="kl",unbalanced_solver="mm",
                                                                      alpha=alpha,M=M,log=True)
            cost=log['fugw_cost']
            
            new_planes.append(gamma)
            new_losses.append(cost)
            # new_plans.append(mapping.pi)
            # new_losses.append(
            #     (
            #         mapping.loss,
            #         mapping.loss_steps,
            #         mapping.loss_times,
            #     )
            # )

        return new_plans, new_losses

    def fit(
        self,
        weights_list,
        features_list,
        geometry_list,
        barycenter_size=None,
        init_barycenter_weights=None,
        init_barycenter_features=None,
        init_barycenter_geometry=None,
        solver="sinkhorn",
        solver_params={},
        nits_barycenter=5,
        device="auto",
        callback_barycenter=None,
        verbose=False,
    ):
        """Compute barycentric features and geometry
        minimizing FUGW loss to list of distributions given as input.
        In this documentation, we refer to a single distribution as
        an a subject's or an individual's distribution.

        Parameters
        ----------
        weights_list (list of np.array): List of weights. Different individuals
            can have weights with different sizes.
        features_list (list of np.array): List of features. Individuals should
            have the same number of features n_features.
        geometry_list (list of np.array or np.array): List of kernel matrices
            or just one kernel matrix if it's shared across individuals
            barycenter_size (int, optional): Size of computed
            barycentric features and geometry. Defaults to None.
        init_barycenter_weights (np.array, optional): Distribution weights
            of barycentric points. If None, points will have uniform
            weights. Defaults to None.
        init_barycenter_features (np.array, optional): np.array of size
            (barycenter_size, n_features). Defaults to None.
        init_barycenter_geometry (np.array, optional): np.array of size
            (barycenter_size, barycenter_size). Defaults to None.
        device: "auto" or torch.device
            if "auto": use first available gpu if it's available,
            cpu otherwise.
        callback_barycenter: callable or None
            Callback function called at the end of each barycenter step.
            It will be called with the following arguments:

                - locals (dictionary containing all local variables)

        Returns
        -------
        barycenter_weights: np.array of size (barycenter_size)
        barycenter_features: np.array of size (barycenter_size, n_features)
        barycenter_geometry: np.array of size
            (barycenter_size, barycenter_size)
        plans: list of arrays
        duals: list of (array, array)
        losses_each_bar_step: list such that l[s][i]
            is a tuple containing:
                - loss
                - loss_steps
                - loss_times
            for individual i at barycenter computation step s
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            else:
                device = torch.device("cpu")

        if barycenter_size is None:
            barycenter_size = weights_list[0].shape[0]

        # Initialize barycenter weights, features and geometry
        if init_barycenter_weights is None:
            barycenter_weights = (
                torch.ones(barycenter_size) / barycenter_size
            ).to(device)
        else:
            barycenter_weights = _make_tensor(
                init_barycenter_weights, device=device
            )

        if init_barycenter_features is None:
            barycenter_features = torch.ones(
                (features_list[0].shape[0], barycenter_size)
            ).to(device)
            barycenter_features = barycenter_features / torch.norm(
                barycenter_features, dim=1
            ).reshape(-1, 1)
        else:
            barycenter_features = _make_tensor(
                init_barycenter_features, device=device
            )

        if init_barycenter_geometry is None and self.learn_geometry is False:
            raise ValueError(
                "In the fixed support case, init_barycenter_geometry must be"
                " provided."
            )
        elif init_barycenter_geometry is None and self.learn_geometry is True:
            barycenter_geometry = (
                torch.ones((barycenter_size, barycenter_size)).to(device)
                / barycenter_size
            )
        else:
            barycenter_geometry = _make_tensor(
                init_barycenter_geometry, device=device
            )

        plans = None
        duals = None
        losses_each_bar_step = []

        for idx in range(nits_barycenter):
            if verbose:
                console.log(
                    f"Barycenter iterations {idx + 1} / {nits_barycenter}"
                )
            # Transport all elements
            plans, losses = self.compute_all_ot_plans(
                plans,
                duals,
                weights_list,
                features_list,
                geometry_list,
                barycenter_weights,
                barycenter_features,
                barycenter_geometry,
                solver,
                solver_params,
                device,
                verbose,
            )

            losses_each_bar_step.append(losses)

            # Update barycenter features and geometry
            barycenter_features = self.update_barycenter_features(
                plans, features_list, device
            )
            if self.learn_geometry:
                barycenter_geometry = self.update_barycenter_geometry(
                    plans, weights_list, geometry_list, self.force_psd, device
                )

            if callback_barycenter is not None:
                callback_barycenter(locals())

        return (
            barycenter_weights,
            barycenter_features,
            barycenter_geometry,
            plans,
            duals,
            losses_each_bar_step,
        )
        
