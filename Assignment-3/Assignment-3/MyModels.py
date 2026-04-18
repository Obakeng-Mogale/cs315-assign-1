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
    def __init__(self, n_clusters, max_iter=100, tol= 1e-6):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.Data = None
# 2. Placeholders for learned parameters (M-Step updates)
        self.means = None         # Will be shape (k, d) or (d, k) depending on your setup
        self.covariances = None   # Will be shape (k, d, d)
        self.coeff_ = None        # Mixing coefficients (pi), shape (k,)
        
        # 3. Placeholders for latent variables and tracking (E-Step updates)
        self.responsibilities_ = None  # The gamma matrix, shape (N, k)
        self.log_likelihoods_ = []     # List to track convergence
        
        # (Optional) Store data, though usually this is passed directly to fit(X)
        self.Data = None
        return

    def fit(self, X):
        self.d, self.N = X.shape
        self.Data = X

        """ running kmeans for initialization"""

        km = Kmeans(n_clusters = self.n_clusters)
        km.fit(self.Data, max_iter = self.max_iter, tol = self.tol)

        """ initializing the parameters"""
        self.means = km.cluster_centers_
        self.coeff_ = km.counts_/self.N
        # Create empty 3D array
        self.covariances = np.zeros((self.n_clusters, self.d, self.d))


        """initialize the covariance matrix"""
        reg_cov = 1e-6 * np.eye(self.d)
        for k in range(self.n_clusters):
            # Find points assigned to cluster k by K-Means
            cluster_points = X[:, km.labels_[:, k] == 1]
            if cluster_points.shape[1] > 1:
                # Add regularization here too
                cov_k = np.cov(cluster_points) + reg_cov
            else:
                cov_k = np.eye(self.d) # Fallback if cluster is empty
            self.covariances[k] = cov_k

        """ perform the EM algorithm """
        for i in range(self.max_iter):
            
            # Expectation step
            self._e_step(X)
            
            # Maximization step
            self._m_step(X)
            
            # Check Convergence
            log_lik = self.compute_loglik(X)
            self.log_likelihoods_.append(log_lik)
            break_bool = False
            if i > 0:
                shift = abs(self.log_likelihoods_[-1] - self.log_likelihoods_[-2])
                if shift < self.tol:
                    print(f"GMM Converged at iteration {i}")
                    break

            if break_bool:
                break
        return

    def predict(self, X_new):
        """
        Predict the cluster index for new data X_new of shape (d, m).
        """
        m = X_new.shape[1]
        weighted_probs = np.zeros((m, self.n_clusters))
        
        for j in range(self.n_clusters):
            pdf_values = self.normal_(X_new, self.means[:, j], self.covariances[j])
            weighted_probs[:, j] = self.coeff_[j] * pdf_values
            
        # Return the index of the cluster with the maximum probability/responsibility
        return np.argmax(weighted_probs, axis=1)

    def _e_step(self, X):
        """
        Calculates the responsibilities (gamma) for all data points.
        X shape: (d, N)
        """
        # 1. Initialize an array to hold the numerators (weighted probabilities)
        # Shape: (N, K)
        weighted_probs = np.zeros((self.N, self.n_clusters))
        
        # 2. Calculate the numerator for each cluster (this loop replaces your nested loops!)
        for j in range(self.n_clusters):
            # Get the PDF values for all N data points for cluster j
            # Assuming self.normal_ handles X as (d, N) and returns an array of shape (N,)
            pdf_values = self.normal_(X, self.means[:, j], self.covariances[j])
            
            # Multiply by the mixing coefficient pi_j and store in the j-th column
            weighted_probs[:, j] = self.coeff_[j] * pdf_values
            
        # 3. Calculate the denominator for each data point
        # Sum across the clusters (axis=1 sums across the columns) -> Shape becomes (N,)
        sum_probs = np.sum(weighted_probs, axis=1)
        
        # 4. Normalize to get the final responsibilities
        # We add [:, np.newaxis] to reshape sum_probs to (N, 1) so it divides the (N, K) matrix correctly.
        # We also add a tiny 1e-15 to prevent completely empty clusters from causing a "divide by zero" error.
        self.responsibilities_ = weighted_probs / (sum_probs[:, np.newaxis])

    def _m_step(self, X):
        N_j = np.sum(self.responsibilities_, axis = 0) 
        self.means = (self.Data @ self.responsibilities_)/N_j 
        self.coeff_ = N_j/self.N

        # Define a small regularization term (epsilon)
        reg_cov = 1e-6 * np.eye(self.d)

        for j in range(self.n_clusters):
            diff = X - self.means[:, j][:, np.newaxis] 
    
            # 2. Get the responsibilities for THIS cluster and reshape to (N, 1)
            weights = self.responsibilities_[:, j] 
    
            # 3. Apply the weights to the differences
            weighted_diff = diff * weights # Shape: (N, d)
    
            # 4. Matrix multiply the transposed weighted diffs by the original diffs
            cov_j = (weighted_diff @ diff.T) / N_j[j]
            
            # 5. ADD REGULARIZATION HERE to prevent singular matrices!
            cov_j += reg_cov
            
            # 6. Store it
            self.covariances[j] = cov_j

    def compute_loglik(self, X):
        """
        Computes the log-likelihood of the data given the current GMM parameters.
        X shape: (d, N)
        """
        # 1. Initialize an array to hold the weighted probabilities for each point and cluster
        # Shape will be (N, K)
        weighted_probs = np.zeros((self.N, self.n_clusters))
        
        # 2. Calculate pi_k * N(x | mu_k, Sigma_k) for all points, cluster by cluster
        for j in range(self.n_clusters):
            # Assuming self.normal_ returns a 1D array of shape (N,) 
            # containing the PDF values for all N points evaluated at cluster j's parameters
            pdf_values = self.normal_(X, self.means[:, j], self.covariances[j])
            
            # Multiply by the mixing coefficient pi_j
            weighted_probs[:, j] = self.coeff_[j] * pdf_values
            
        # 3. Sum the probabilities across all clusters (axis=1 sums the rows)
        # Shape becomes (N,)
        sum_probs = np.sum(weighted_probs, axis=1)
        
        # 4. Take the natural log of the sums
        # Note: We add a tiny epsilon (1e-15) to prevent log(0) errors if a probability is extremely small
        log_probs = np.log(sum_probs)
        
        # 5. Sum the log-probabilities over all N data points
        total_log_likelihood = np.sum(log_probs)
        
        return total_log_likelihood

    def normal_(self, X, mean, cov):
        d, N = X.shape
        diff = X - mean[:, np.newaxis]
        sign, log_det = np.linalg.slogdet(cov)
        inv_cov = np.linalg.inv(cov)
        mahalanobis = np.sum(diff * (inv_cov @ diff), axis=0)
        log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)
        return np.exp(log_prob)