from mhi import *
import math
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def mhiHelper(frames, mhiClass, theta, tau, fps):
    f = presetImage(frames[0])
    presetedFrames = [f]
    bts =  np.zeros(np.shape(f), dtype=np.float64)
    mhis = np.zeros(np.shape(f), dtype=np.float64)
    
    for i in range(1, min(int(tau*fps),len(frames))):
        f = presetImage(frames[i])
        presetedFrames.append(f)
        bt = mhiClass.Bt(f, presetedFrames[i-1])
        bts+=bt
        mhis = mhiClass.MHIs(bt, mhis)
    bts[bts>1] = 1.
    #plt.imshow(mhis)
    return mhis, bts

def video_writer(filename, frame_size, fps):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break
    video.release()
    yield None

def getFrames(filePathName):
    image_gen = video_frame_generator(filePathName + ".avi")
    video = cv2.VideoCapture(filePathName + ".avi")
    ima = image_gen.__next__()
    frameList = []
    while ima is not None:
        frameList.append(ima)
        ima = image_gen.__next__()  
    return frameList, video.get(cv2.CAP_PROP_FPS)


class trainPredictResult:
    def __init__(self):
        self.trainPercentage = 0.7
        self.thetas = {
            'boxing': 15,
            'handclapping': 15,
            'handwaving': 10,
            'jogging': 25,
            'running': 35,
            'walking': 30
        }
        self.actionsList = ['boxing','handclapping','handwaving','jogging','running','walking']
        self.taus = {
            'boxing': 5.,
            'handclapping': 5.,
            'handwaving': 5.,
            'jogging': 13.,
            'running': 13.,
            'walking': 5.
        }
        self.defaultTau = 5
        self.names = None
        self.actions = None
        self.frames = None
        self.X = None
        self.Y = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.model = neighbors.KNeighborsClassifier()
        self.predictY = None
        
    def experiment(self, filePathName):
        video = cv2.VideoCapture(filePathName + ".avi")
        image_gen = video_frame_generator(filePathName + ".avi")
        ima = image_gen.__next__()
        fps = video.get(cv2.CAP_PROP_FPS)
        writer = video_writer(filePathName + "_out.avi", (ima.shape[1], ima.shape[0]), fps)
        writer.write(ima)
        frameList = []
        while ima is not None:
            frameList.append(ima)
            ima = image_gen.__next__()
        print(len(frameList))
        count = 0
        temp = min(25, len(frameList))
        print(temp)
        predictingFrames = frameList[count:temp]
        print(len(predictingFrames))
        actionLast = ''
        while len(predictingFrames)>0:
            mhiclass = MHI(21, self.defaultTau)
            mhi,_ = mhiHelper(predictingFrames, mhiclass, 21, self.defaultTau, 1000)
            hu = HU(mhi).huMoments()
            print(hu)
            action = None
            try:
                action = self.actionsList[int(self.model.predict([hu]))]
            except:
                pass
            if action == None:
                action = actionLast
            actionLast = action
            print(action)
            
            temp = 25+1
            
            for i in range(temp):
                if count+i < len(frameList):
                    f = frameList[count+i]
                    if action!='':
                        f = cv2.putText(f, action, (int(f.shape[0]/2),int(f.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0,0,255), 1, cv2.LINE_AA)
                    writer.write(f)    
                else: 
                    break
            count += temp
            predictingFrames = frameList[count:int(min(25+1+count, len(frameList)))]
            print(count)
            print(int(min(25+1+count, len(frameList))))
            print(len(frameList))
            print(len(predictingFrames))
            
        writer.release()
        
    def experiment1(self, filePathName):
        video = cv2.VideoCapture(filePathName + ".avi")
        image_gen = video_frame_generator(filePathName + ".avi")
        ima = image_gen.__next__()
        fps = video.get(cv2.CAP_PROP_FPS)
        writer = video_writer(filePathName + "_out.avi", (ima.shape[1], ima.shape[0]), fps)
        writer.write(ima)
        frameList = []
        while ima is not None:
            frameList.append(ima)
            ima = image_gen.__next__()
        mhiclass = MHI(21, self.defaultTau)
        mhi,_ = mhiHelper(frameList[:25], mhiclass, 21, self.defaultTau, 1000)
        hu = HU(mhi).huMoments()
        print(hu)
        action = self.actionsList[int(self.model.predict([hu]))]
        print(action)
            
        for f in frameList:
            f = cv2.putText(f, action, (int(f.shape[0]/2),int(f.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0,0,255), 1, cv2.LINE_AA)
            writer.write(f)    
            
        writer.release()
        
    def prepareSequence(self):
        # sequence file downloaded via https://www.csc.kth.se/cvap/actions/
        # renamed 'cq1'
        file1 = open('cq1.txt', 'r')
        Lines = file1.readlines()
        names = []
        actions = []
        frames  = []
        for line in Lines:
            if line[0:6] == 'person':
                j = 0
                i = 0
                for j in range(0, len(line)):
                    if line[j:j+1] == "_":
                        break
                k = j
                for k in range(j+1, len(line)):
                    if line[k:k+1] == "_":
                        break

                i = k+3
                #print(line[0:i])
                names.append(line[0:i])
                actions.append(line[j+1:k])
                for j in range(i, len(line)):
                    if line[j:j+1] == 'f':
                        break
                _frame = []
                while j < len(line)-1:
                    _start = ''
                    _end = ''
                    for i in range(j, len(line)):
                        try:
                            int(line[i:i+1])
                            break
                        except:
                            pass
                    for j in range(i, len(line)):
                        if line[j:j+1] == '-':
                            break
                        else:
                            _start += line[j:j+1]
                    for i in range(j+1,len(line)):
                        try:
                            int(line[i:i+1])
                            break
                        except:
                            pass
                    for j in range(i, len(line)):
                        if line[j:j+1] == ',':
                            break
                        else:
                            _end += line[j:j+1]
                    try:
                        _frame.append([int(_start), int(_end)])
                    except:
                        pass
                frames.append(_frame)
        self.names = names
        self.actions = actions
        self.frames = frames
        
    def setupData(self):
        x = []
        y = []
        print("processing " + str(len(self.names)))
        count = 1
        for i in range(len(self.names)):
            print(count)
            image_gen = video_frame_generator("video/" + self.names[i] + "_uncomp.avi")
            video = cv2.VideoCapture("video/" + self.names[i] + "_uncomp.avi")
            fps = video.get(cv2.CAP_PROP_FPS)
            frameList = []
            ima = image_gen.__next__()
            while ima is not None:
                frameList.append(ima)
                ima = image_gen.__next__()
            for f in self.frames[i]:
                f0 = f[0]
                f1 = f[1]
                while f0<=f1-25:
                    mhiclass = MHI(theta = self.thetas[self.actions[i]], tau = self.taus[self.actions[i]])
                    mhi, _ = mhiHelper(frameList[f[0]:f[0]+25], mhiclass, self.thetas[self.actions[i]], self.taus[self.actions[i]], 10000)
                    hu = HU(mhi).huMoments()
                    if math.isnan(hu[0]):
                        pass
                    else:
                        x.append(hu)
                        y.append(float(self.actionsList.index(self.actions[i])))
                    f0 += 25
        self.X = x
        self.Y = y
        return x, y
    
    def setupData1(self):
        x = []
        y = []
        print("processing " + str(len(self.names)))
        count = 1
        for i in range(len(self.names)):
            print(count)
            image_gen = video_frame_generator("video/" + self.names[i] + "_uncomp.avi")
            video = cv2.VideoCapture("video/" + self.names[i] + "_uncomp.avi")
            fps = video.get(cv2.CAP_PROP_FPS)
            frameList = []
            ima = image_gen.__next__()
            while ima is not None:
                frameList.append(ima)
                ima = image_gen.__next__()
            for f in self.frames[i]:
                mhiclass = MHI(theta = self.thetas[self.actions[i]], tau = self.taus[self.actions[i]])
                mhi, _ = mhiHelper(frameList[f[0]:f[1]], mhiclass, self.thetas[self.actions[i]], self.taus[self.actions[i]], 10000)
                hu = HU(mhi).huMoments()
                if math.isnan(hu[0]):
                    pass
                else:
                    x.append(hu)
                    y.append(float(self.actionsList.index(self.actions[i])))
        self.X = x
        self.Y = y
        return x, y
        
    def splitData(self):
        data = []
        count = len(self.X)
        for i in range(count):
            data.append([self.X[i], self.Y[i]])
        data = np.array(data)
        np.random.shuffle(data)
        trainQty = int(count * self.trainPercentage)
        trained = data[0:trainQty]
        tested = data[trainQty: count]
        self.trainX = []
        self.trainY = []
        for d in trained:
            self.trainX.append(d[0])
            self.trainY.append(d[1])
        self.testX = []
        self.testY = []
        for d in tested:
            self.testX.append(d[0])
            self.testY.append(d[1])
        
    def trainModel(self):
        # train with knn
        
        #print(self.trainX)
        #print(self.trainY)
        self.model.fit(self.trainX, self.trainY)
        
        
    def predict(self):
        self.predictY = self.model.predict(self.testX)
        
    def valuation(self):
        self.predictV = self.model.predict(self.trainX)    
        
    def predictResult(self):
        correct = 0
        for i in range(len(self.predictY)):
            if self.predictY[i] == self.testY[i]:
                correct+=1
        print(float(correct)/len(self.predictY))
        return float(correct)/len(self.predictY)  
    
    def predictResultV(self):
        correct = 0
        for i in range(len(self.predictV)):
            if self.predictV[i] == self.trainY[i]:
                correct+=1
        print(float(correct)/len(self.predictV))
        return float(correct)/len(self.predictV)  
    
    def confusionMatrix(self):
        cm_train = confusion_matrix(self.testY, self.predictY,labels = self.actionsList)
        plt.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Reds)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(6, self.actionsList , rotation=45)
        plt.yticks(6, self.actionsList)

        fmt = '.2f' 
        thresh = cm_train.max() / 2.
        for i, j in itertools.product(range(cm_train.shape[0]), range(cm_train.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
if __name__ == "__main__":
    tpr = trainPredictResult()
    tpr.prepareSequence()
    tpr.setupData1()
    tpr.splitData()
    tpr.trainModel()
    tpr.predict()
    tpr.valuation()
    #tpr.confusionMatrix()
    #tpr.experiment1("video/t1")
    
        