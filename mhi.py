import numpy as np
import cv2

class MHI:
    def __init__(self, theta, tau):
        self.theta = theta
        self.tau = tau
        
    def Bt(self, image, previousImage):
        image = toFloat(image)
        previousImage = toFloat(previousImage)
        dif = np.abs(image - previousImage)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if dif[i][j] >= self.theta:
                    dif[i][j] = 1.
                else:
                    dif[i][j] = 0.
        
        return dif
    
    def MHIs(self, bt, previousMHI):
        mhis = np.zeros(np.shape(bt), dtype=np.float64)
        for i in range(bt.shape[0]):
            for j in range(bt.shape[1]):
                if bt[i][j] >= 1.:
                    mhis[i][j] = self.tau
                else:
                    mhis[i][j] = max(previousMHI[i][j]-1.,0.)
        return mhis
    
class HU:
    def __init__(self, image):
        #<pq> ∈ {20, 11, 02, 30, 21, 12, 03, 22}
        self.image = image
        self.pq = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
        self.mu = [0.,0.,0.,0.,0.,0.,0.]
        self.v = [0.,0.,0.,0.,0.,0.,0.]
        self.shape = np.shape(image)
        self.x, self.y = None, None
        self.x_, self.y_ = None, None
        self.M10 = None
        self.M01 = None
        self.M00 = None
        self.mu00 = None
        
    def calculateM(self, x, y, i, j, image):
        return np.sum(pow(x, i) * pow(y, j) * image)
    
    def initialize(self):
        self.x, self.y = np.meshgrid(range(self.shape[1]), range(self.shape[0]))
        self.M10 = toFloat(self.calculateM(self.x, self.y, 1, 0, self.image))
        self.M01 = toFloat(self.calculateM(self.x, self.y, 0, 1, self.image))
        self.M00 = toFloat(self.calculateM(self.x, self.y, 0, 0, self.image))
        self.x_ = self.M10/self.M00
        self.y_ = self.M01/self.M00
        self.mu00 = self.centralMoments(0,0)
        
    def centralMoments(self, p, q):
        xDif = self.x - self.x_
        yDif = self.y - self.y_
        return np.sum(pow(xDif, p) * pow(yDif, q) * self.image)
    
    def scaledMoments(self,muPQ, mu00, p, q):
        return muPQ/pow(mu00, (1+(p+q)/2))
        
    def huMoments(self):
        self.initialize()
        for i, (p,q) in enumerate(self.pq):
            self.mu[i] = self.centralMoments(p,q)
            self.v[i] = self.scaledMoments(self.mu[i], self.mu00, p, q)
        mu = self.mu
        hu = [None, None, None, None,None, None]
        hu[0] = mu[0]+mu[2]
        hu[1] = pow((mu[0] - mu[2]) + (4 * pow(mu[1], 2)),2)
        hu[2] = pow(mu[3] - (3*mu[5]), 2) + pow((3*mu[4]) - mu[6], 2)
        hu[3] = pow(mu[3]+mu[5], 2) + pow(mu[4]+mu[6], 2)
        hu[4] = (mu[3] - 3*mu[5]) * (mu[3]+mu[5]) *(pow(mu[3]+mu[5], 2) - 3 * pow(mu[4]+mu[6], 2)) + (3*mu[4]-mu[6])*(mu[4]+mu[6])*(3*pow(mu[3]+mu[5], 2) - pow(mu[4]+mu[6], 2))
        hu[5] = (mu[0] - mu[2])*(pow(mu[3] + mu[5], 2)-pow(mu[4]+mu[6], 2)) + 4 * mu[1] *(mu[3] + mu[5])*(mu[4]+mu[6])
        return np.concatenate((np.array(hu), np.array(mu)))
    
def toFloat(obj):
    return obj.astype(np.float32)
        
def presetImage(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gBlur = cv2.GaussianBlur(grayFrame, (15,15), 11)
    return gBlur