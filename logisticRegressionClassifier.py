import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class LogisticRegression(object):
    #weights : 1d-array, shape = [1, 1 + n_features], 1x3 Weights after training
    def __init__(self, learningRate, numIterations = 10, penalty = 'L2', C = 0.01):
        self.learningRate = learningRate          #default 0.01.
        self.numIterations = numIterations
        self.penalty = penalty
        self.C = C

    def train(self, X_train, y_train, tol = 10 ** -4):
        """
        Parameters
        -----------
        X_train : {array-like}, shape = [n_samples, n_features]
        y_train : array-like, shape = [n_samples], values = 1|0, Labels (target values).
        tol : float, optional
            Giá trị cho biết sự thay đổi giữa các epochs. Mặc định là 10 ** -4
        Returns:
        -----------
        self : object

        """
        tolerance = tol * np.ones([1, np.shape(X_train)[1] + 1]) #1x3
        self.weights = np.zeros(np.shape(X_train)[1] + 1)  #3x1
        X_train = np.c_[np.ones([np.shape(X_train)[0], 1]), X_train] #X_train cũ:489x2 chuyển thành 489x3, cột ones ở đầu
        self.costs = []                  #Giá trị của hàm chi phí cho mỗi lần lặp lại giảm dần độ dốc

        for i in range(self.numIterations):
            
            z = np.dot(X_train, self.weights)
            errors = y_train - logistic_func(z)
            if self.penalty is not None:                
                delta_w = self.learningRate * (self.C * np.dot(errors, X_train) + np.sum(self.weights))  
            else:
                delta_w = self.learningRate * np.dot(errors, X_train)
                
            self.iterationsPerformed = i
            #so lan lap lai cua giam doc duoc thuc hien truoc khi dat den tolerance level

            if np.all(abs(delta_w) >= tolerance):

                #weight update
                self.weights += delta_w

                #Costs
                if self.penalty is not None:
                    self.costs.append(reg_logLiklihood(X_train, self.weights, y_train, self.C))
                else:
                    self.costs.append(logLiklihood(z, y_train))
            else:
                break
            
        return self
                    
    def predict(self, X_test, pi = 0.5):
        """
        pi : float, probability, optional
            Ngưỡng xác suất để dự đoán positive class. Mặc định là 0,5.
        """
        z = self.weights[0] + np.dot(X_test, self.weights[1:])        
        probs = np.array([logistic_func(i) for i in z])
        predictions = np.where(probs >= pi, 1, 0)
        return predictions, probs
        
    def performanceEval(self, predictions, y_test):

        TP, TN, FP, FN, P, N = 0, 0, 0, 0, 0, 0
        
        for idx, test_sample in enumerate(y_test):
            
            if predictions[idx] == 1 and test_sample == 1:
                TP += 1       
                P += 1
            elif predictions[idx] == 0 and test_sample == 0:                
                TN += 1
                N += 1
            elif predictions[idx] == 0 and test_sample == 1:
                FN += 1
                P += 1
            elif predictions[idx] == 1 and test_sample == 0:
                FP += 1
                N += 1
            
        accuracy = (TP + TN) / (P + N)                
        sensitivity = TP / P      #recall
        PPV = TP / (TP + FP)      #precision
        F1 = (2*PPV*sensitivity)/(PPV+sensitivity)

        performance = {'Accuracy': accuracy,
                       'Recall': sensitivity,
                       'F1-score': F1,
                       'Precision': PPV}
      
        return performance
        
    def predictionPlot(self, X_test, y_test):
        #plot của các mẫu test được ánh xạ lên hàm logistic.
        zs = self.weights[0] + np.dot(X_test, self.weights[1:])        
        probs = np.array([logistic_func(i) for i in zs])
        
        plt.figure()
        plt.plot(np.arange(-10, 10, 0.1), logistic_func(np.arange(-10, 10, 0.1)))        
        colors = ['r','b']
        probs = np.array(probs)
        for idx,cl in enumerate(np.unique(y_test)):
            plt.scatter(x = zs[np.where(y_test == cl)[0]], y = probs[np.where(y_test == cl)[0]], alpha = 0.8, c = colors[idx], marker = 'o', label = cl, s = 30)

        plt.xlabel('z')
        plt.ylim([-0.1, 1.1])
        plt.axhline(0.0, ls = 'dotted', color = 'k')
        plt.axhline(1.0, ls = 'dotted', color = 'k')
        plt.axvline(0.0, ls = 'dotted', color = 'k')
        plt.ylabel('$\phi (z)$')
        plt.legend(loc = 'upper left')
        plt.title('Logistic Regression Prediction Curve')
        plt.show()
        


    def plotDecisionRegions(self, X_test, y_test, pi = 0.5, res = 0.01):
        """
        Trực quan hóa ranh giới quyết định của bộ phân loại hồi quy logistic được train.

        pi : float, cut-off probability, optional, Ngưỡng xác suất để dự đoán positive class. Mặc định là 0,5.

        res : float, Resolution của contour grid

        """
        x = np.arange(min(X_test[:,0]) - 1, max(X_test[:,0]) + 1, 0.01)
        y = np.arange(min(X_test[:,1]) - 1, max(X_test[:,1]) + 1, 0.01)
        xx, yy = np.meshgrid(x, y, indexing = 'xy')

        data_points = np.transpose([xx.ravel(), yy.ravel()])
        preds, probs = self.predict(data_points, pi)

        colors = ['r','b']
        probs = np.array(probs)

        for idx,cl in enumerate(np.unique(y_test)):
            plt.scatter(x = X_test[:,0][np.where(y_test == cl)[0]], y = X_test[:,1][np.where(y_test == cl)[0]],
                    alpha = 0.8, c = colors[idx],
                    marker = 'o', label = cl, s = 30)

        preds = preds.reshape(xx.shape)
        plt.contourf(xx, yy, preds, alpha = 0.3)
        plt.legend(loc = 'best')
        plt.xlabel('$x_1$', size = 'x-large')
        plt.ylabel('$x_2$', size = 'x-large')


def logistic_func(z):
    return 1 / (1 + np.exp(-z))  
    
def logLiklihood(z, y):
    return -1 * np.sum((y * np.log(logistic_func(z))) + ((1 - y) * np.log(1 - logistic_func(z))))
    
def reg_logLiklihood(x, weights, y, C):
    """

    Parameters
    -----------
    x : {array-like}, shape = [n_samples, n_features + 1]
        Note, first column of x must be a vector of ones.

    weights : 1d-array, shape = [1, 1 + n_features]

    C : float
        Regularization parameter. C = 1/lambda

    """
    z = np.dot(x, weights)

    #L1:
    #reg_term = 1 / 2 * math.sqrt(np.dot(weights.T, weights))
    reg_term = 1 / 2 * np.dot(weights.T, weights)
    return C * (-1 * np.sum((y * np.log(logistic_func(z))) + ((1 - y) * np.log(1 - logistic_func(z))))) + reg_term

