import numpy as np

import numpy as np

class Kmeans:
    def __init__(self, n_clusters: int, random_choice=True):
        self.num_clusters = n_clusters
        self.random_choice = random_choice
        self.cluster_centers_ = None 
        self.labels_ = None
        self.j_history = []
        self.sq_dist = None

    def fit(self, X, max_iter=1000, tol=1e-4):
        self.X = X
        self.d, self.N = X.shape
        
        # Initialize centers as (d, k)
        self.cluster_centers_ = np.zeros((self.d, self.num_clusters))
        
        # Placeholder for binary_split if you implement it later
        if self.random_choice:
            self.rand_choice()
        else:
            print("Binary split not implemented; defaulting to random choice.")
            self.rand_choice()

        for i in range(max_iter):
            old_centers = self.cluster_centers_.copy()

            # E-Step: Vectorized distance and label assignment
            current_j = self.distance()
            self.j_history.append(current_j)

            # M-Step: Vectorized center update
            self.update_centers()

            # Check for convergence
            shift = np.linalg.norm(self.cluster_centers_ - old_centers)
            if shift < tol:
                print(f"Convergence at iteration: {i} (Shift: {shift:.6f})")
                break
        return 

    def rand_choice(self):
        """Select random columns from X as initial centers"""
        random_indices = np.random.choice(self.N, self.num_clusters, replace=False)
        # Fix: Access the column [:, i] instead of the row [i]
        for i in range(self.num_clusters):
            self.cluster_centers_[:, i] = self.X[:, random_indices[i]] 

    def distance(self):
        """Vectorized distance calculation and hard assignment"""
        # 1. Broadcasting: (d, N, 1) - (d, 1, k) -> (d, N, k)
        diff = self.X[:, :, np.newaxis] - self.cluster_centers_[:, np.newaxis, :]
        
        # 2. Squared Euclidean distance summed along feature axis (0) -> (N, k)
        self.sq_dist = np.sum(diff**2, axis=0)
        
        # 3. Find closest cluster for each of the N samples
        best_clusters = np.argmin(self.sq_dist, axis=1) 
        
        # 4. Objective function J (sum of minimum distances)
        total_j = np.sum(np.min(self.sq_dist, axis=1))
        
        # 5. One-hot labels (N x k)
        self.labels_ = np.zeros((self.N, self.num_clusters))
        self.labels_[np.arange(self.N), best_clusters] = 1
        
        return total_j

    def update_centers(self):
        """Vectorized update: averages assigned points for each cluster"""
        # 1. Sum up data points per cluster: (d, N) @ (N, k) -> (d, k)
        cluster_sums = self.X @ self.labels_
        self.counts_ = np.sum(self.labels_, axis=0)
        
        # 2. Count points in each cluster: (k,)
        counts = np.sum(self.labels_, axis=0)
        
        # 3. Compute new means, handling the empty cluster edge case
        for k in range(self.num_clusters):
            if counts[k] > 0:
                self.cluster_centers_[:, k] = cluster_sums[:, k] / counts[k]
            else:
                # Re-initialize empty cluster to a random data point
                self.cluster_centers_[:, k] = self.X[:, np.random.choice(self.N)]

    def predict(self, X_new):
        """Predict cluster index for new d x m data"""
        diff = X_new[:, :, np.newaxis] - self.cluster_centers_[:, np.newaxis, :]
        sq_distances = np.sum(diff**2, axis=0)
        return np.argmin(sq_distances, axis=1)


class GMM:
    def __init__(self):
        return

    def fit(self):
        return

    def predict(self):
