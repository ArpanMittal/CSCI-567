import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
       '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        '''

       assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
       np.random.seed(42)
       N, D = x.shape

       # TODO:
       # - comment/remove the exception.
       # - Initialize means by picking self.n_cluster from N data points
       # - Update means and membership until convergence or until you have made self.max_iter updates.
       # - return (means, membership, number_of_updates)

       # DONOT CHANGE CODE ABOVE THIS LINE
       # center_dist = list()
       # r = np.zeros((N,self.n_cluster), dtype=int)
       # ini_center = np.random.choice(N, self.n_cluster, True)
       # j = np.power(10,10)
       # centers = x[ini_center]
       #
       # iter = 0
       # while iter < self.max_iter:
       #     cluser_sum = [([np.linalg.norm(x2 - center, ord=2) for center in centers])for x2 in x]
       #     # cluster = (([(np.argmin([np.linalg.norm(x2 - center, ord=2) for center in centers])) for x2 in x]))
       #     cluster = (np.argmin(cluser_sum,1))
       #     for idx,val in enumerate(cluster):
       #         r[idx][val] = 1
       #     j_new = np.sum(np.amin(cluser_sum,1))/N
       #     if(np.absolute(j_new-j) < self.e):
       #         break
       #     else:
       #         j = j_new
       #
       #     centers = np.array([np.mean(x[cluster == k], axis=0) for k in range(self.n_cluster)])
       #     iter += 1
       #
       #     # print(r)
       #
       # return (centers,cluster, iter)




       cluster = x[np.random.choice(N, self.n_cluster, replace=True), :]
       # r = np.zeros(N, dtype=int)
       J = np.power(10,10)
       iter = 0
       # r = np.arange(200)
       # temp =
       # temp = mu - np.expand_dims(x, axis=0)
       # temp2 = x- np.expand_dims(mu, axis=1)
       while iter < self.max_iter:

           euclidean = np.sum(((x - np.expand_dims(cluster, axis=1)) ** 2), axis=2)
           r = np.argmin(euclidean, axis=0)

           J_new = np.sum([np.sum((x[r == k] - cluster[k]) ** 2) for k in range(self.n_cluster)]) / N
           if np.absolute(J - J_new) <= self.e:
               break
           J = J_new

           cl_new = np.array([np.mean(x[r == k], axis=0) for k in range(self.n_cluster)])
           index = np.where(np.isnan(cl_new))
           cl_new[index] = cluster[index]
           cluster = cl_new
           iter += 1
       return (cluster, r, iter)
       # print(ini_center)
       # raise Exception(
       #     'Implement fit function in KMeans class (filename: kmeans.py)')
       # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels
        centroid_labels = []
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, i = k_means.fit(x)
        # self.centroids = centroids
        temp = np.array([y[membership == i] for i in range(self.n_cluster)])
        for list in temp:
            if len(list) ==0:
                centroid_labels.append(0)
            else:
                centroid_labels.append(np.asscalar(np.argmax(np.bincount(list))))
        centroid_labels =  np.array(centroid_labels)
        # index = np.where(np.isnan(temp))
        # temp[index] = 0
        # print(temp)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeansClassifier class (filename: kmeans.py)')

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        temp2 = np.expand_dims(self.centroids, axis=1)
        diff = np.argmin(np.sum((x - temp2) ** 2, axis=2), axis=0)
        labels =  self.centroid_labels[diff]
        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement predict function in KMeansClassifier class (filename: kmeans.py)')
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

