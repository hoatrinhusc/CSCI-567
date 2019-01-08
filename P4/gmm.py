import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.means, r, dummy = KMeans(self.n_cluster, self.max_iter, self.e).fit(x)
            variances = np.empty((0,D))
            pi_k = []

            for k in range(self.n_cluster):
                   vt = self.means[k] - x
                   x_mu = vt.reshape(N,D,1)
                   mu_x = vt.reshape(N,1,D)
                   gamma = np.array(r == k).reshape(N,1,1)
                   Nk = np.count_nonzero(gamma)
                   variance = np.sum(gamma * x_mu * mu_x, axis = 0)
                   variance /= Nk
                   variances = np.append(variances, variance, axis = 0)
                   pi_k.append(Nk/N) 
            self.variances = variances.reshape(self.n_cluster, D,D)
            self.pi_k = np.array(pi_k)
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            variances = np.empty((0,D))
            self.means = np.random.rand(self.n_cluster, D)
            variance = np.identity(D)
            for k in range(self.n_cluster):
                   variances = np.append(variances, variance, axis = 0)
            self.variances = variances.reshape(self.n_cluster, D,D)
            self.pi_k = np.ones(self.n_cluster) /self.n_cluster
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE

        # Compute log-likelihood  
        l = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
        step = 0
        for step in range(self.max_iter):
             step += 1
             soft_gma = np.empty((0, self.n_cluster))
             for i in range(N):
                   soft_gma_i = np.empty((0,),float)
                   for k in range(self.n_cluster):
                         distribution = self.Gaussian_pdf(self.means[k], self.variances[k])
                         soft_gma_i = np.append(soft_gma_i, self.pi_k[k] * distribution.getLikelihood(x[i]))
                   soft_gma_i = soft_gma_i / np.sum(soft_gma_i)
                   soft_gma = np.append(soft_gma, [soft_gma_i], axis = 0)

             Nk = np.sum(soft_gma, axis = 0)
             xnew = x.reshape(N, 1, D)
             soft_gma = soft_gma.reshape(N, self.n_cluster,1)
             means = np.sum(soft_gma * xnew, axis = 0)

             variances = np.empty((0,D,D),float)
             pi_k = np.empty((0,), float)
        
             for k in range(self.n_cluster):
                   var = np.empty((0,D,D),float)
                   means[k] /= Nk[k]
                   for i in range(N):
                          vt = means[k] - x[i]
                          var = np.append(var, [soft_gma[i][k] * vt.reshape(D,1) * vt.reshape(1,D)], axis = 0)
 
                   vark = np.sum(var, axis = 0)/Nk[k]
                   variances = np.append(variances, [vark], axis = 0)
                   pi_k = np.append(pi_k, Nk[k]/N)

             self.means = means
             self.variances = variances.reshape(self.n_cluster, D, D)
             self.pi_k = np.array(pi_k)
             # Compute new log-likelihood  
             lnew = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
             if np.abs(l - lnew) <= self.e:
                break
             l = lnew
        return step


        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        cluster = np.random.choice(self.n_cluster, N, self.pi_k)
        samples = np.empty((0, self.means.shape[1]))
        for i, k in enumerate(cluster):
            mean = self.means[k]
            var = self.variances[k]
            samples = np.append(samples, np.random.multivariate_normal(mean, var), axis = 0)
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        N, D = x.shape
        log_likelihood = 0.

        for i in range(N):
             prob = 0.
             for k in range(self.n_cluster):
                  distribution = self.Gaussian_pdf(means[k], variances[k])
                  prob += float(pi_k[k] * distribution.getLikelihood(x[i]))
             log_likelihood += float(np.log(prob)) 
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            
            #det = np.linalg.det(self.variance)
            D = self.mean.shape[0]
            rank = np.linalg.matrix_rank(self.variance)
            while rank < D:
                    self.variance += np.identity(D) * (10 ** -3)
                    #var += np.eye(D) * 1e-3
                    rank = np.linalg.matrix_rank(self.variance)

            #if det != 0:
            self.inv = np.linalg.inv(self.variance)
            det = np.linalg.det(self.variance)
            self.c = (2 * np.pi) ** D * det
            #else:
            #        while det == 0:
            #                 self.variance += np.identity(D) * (10 ** -3)
            #                 det = np.linalg.det(self.variance)
            #        self.inv = np.linalg.inv(self.variance)
            #        self.c = (2 * np.pi) ** D * det

            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            vt = x - self.mean
            p = np.exp(-0.5 * np.dot(np.dot(vt, self.inv), vt.T))
            p /= np.sqrt(self.c)
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
