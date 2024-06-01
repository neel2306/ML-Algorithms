import numpy as np

class LinearRegression:
    '''
    Custom implementation of the Linear Regression algorithm using numpy only.
    '''
    def __init__(self, lr:int = 0.001, epochs:int = 500, optimizer:str = 'lms') -> None:
        self.lr = lr #Learning rate
        self.epochs = epochs #Number of iterations

        #Initialize weights/parameters and bias.
        self.weights = None
        self.bias = None
        self.optimizer = optimizer

    def fit(self, X, y):
        self.X = X
        self.y = y
        training_examples, num_features = self.X.shape

        #Initialize weight and bias to a random number.
        self.weights = np.random.rand(num_features)  #Will create an array of shape [num_features, 1], initialize that many number of thetas.
        self.bias = 0 #Good practice to initialize biases to 0.

        #Update weights.
        for epoch in range(self.epochs):
            self.LMS(training_examples=training_examples)
        return self

    def LMS(self, training_examples):
        '''
        Function that performs gradient descent on Least Means Square (LMS) cost function
        '''
        y_hat = self.predict(self.X)

        #Compute gradients/partial derivatives.
        dw = (1/training_examples) * np.dot(self.X.T, (y_hat - self.y))
        db = (1/training_examples) * np.sum(y_hat - self.y)

        #Update weights and bias.
        self.weights = self.weights - self.lr * dw
        self.bias = self.bias - self.lr * db
        return self
    
    def normal_equation(self):
        '''
        Computes analytical values of weights.
        weights = ([X.T.X]^-1).X.T.y
        '''
        transpose_x = np.linalg.inv(np.transpose(self.X).dot(self.X))
        transpose_y = np.dot(np.transpose(self.X), self.y)

        #Solve.
        try:
            self.bias, *self.weights = transpose_x.dot(transpose_y)
            return self.weights
        except np.linalg.LinAlgError:
            return None

    def predict(self, X):
        '''
        Function that runs our hypothesis on X to generate output y_hat
        '''
        return np.dot(X, self.weights) + self.bias