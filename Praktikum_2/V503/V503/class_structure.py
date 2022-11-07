import numpy as np

class KNN:
    '''KNN Classifier.

    Attributes
    ----------
    k : int
        Number of neighbors to consider.
    '''
    def __init__(self, k):
        '''Initialization.
        Parameters are stored as member variables/attributes.
        
        Parameters
        ----------
        k : int
            Number of neighbors to consider.
        '''
        self.k = k

    def fit(self, X, y):
        '''Fit routine.
        Training data is stored within object.
        
        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_attributes)
            Training data.
        y : numpy.array shape=(n_samples)
            Training labels.
        '''
        self.y=y
        self.X=X
        


    def predict(self, X):
        '''Prediction routine.
        Predict class association of each sample of X.
        
        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_attributes)
            Data to classify.
        
        Returns
        -------
        prediction : numpy.array, shape=(n_samples)
            Predictions, containing the predicted label of each sample.
        '''
        
        lenX=len(X)
        lenselfX=len(self.X[0,:])
        predictedlabels=np.empty(lenX)
        distances=np.empty(lenselfX)
        differences=np.empty((len(self.X[:,0]),lenselfX))
        for n in range(lenX):                      
            for i in range(lenselfX):
                differences[:,i]=(self.X[:,i]-X[n,i])**2
            distances=np.sqrt(np.sum(differences,axis=1))
            sorted=np.argsort(distances)
            predictedlabels[n]=np.bincount(self.y[sorted[0:self.k]]).argmax()
        return predictedlabels


            
          