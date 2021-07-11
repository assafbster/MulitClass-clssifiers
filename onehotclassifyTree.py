from timeit import default_timer as timer
import addcopyfighandler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.linalg import sqrtm
from ismember import ismember
from typing import List, Any, Iterable
import torchvision
import torchvision.transforms as transforms
#######################################################################

def flatten_recursive(lst: List[Any]) -> Iterable[Any]:
    """Flatten a list using recursion."""
    for item in lst:
        if isinstance(item, list):
            yield from flatten_recursive(item)
        else:
            yield item

#######################################################################
def flattenlist(ll):
    return list(flatten_recursive(ll))

#######################################################################

def toc(prevtime):
    curtime = timer()
    print('Elaped time {:4f}'.format(curtime - prevtime) + ' seconds')
    return curtime

#######################################################################

def formatArray(v):
    vstr = ''
    for jj in range(v.shape[0]):
        vstr = vstr + '{:.4f} '.format(v[jj])
    return vstr

#######################################################################

def plotData(X,y,clf,rownum,classname,testRegret):
    h = .02
    nsamps = y.size
    maxplottedsamps = 50
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first

    # 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    # 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    cm = plt.cm.binary
    cm_bright = ListedColormap(['#FF0000', '#0000FF','#00FF00'])
    colors = ['r','g','b']
    # ax = plt.subplot(1,3,1)
    markerlist = ['2','3','4']
    ProbMat = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    for ii in range(3):
        Z = ProbMat[:,ii].reshape(xx.shape)
        ax = plt.subplot(3, 3, 3*(rownum-1)+ii+1)
        for jj in range(3):
            curinds = np.arange(nsamps)[y==jj]
            ax.scatter(X[curinds, 0][:maxplottedsamps], X[curinds, 1][:maxplottedsamps], color = colors[jj],
                       cmap=cm_bright, edgecolors='k',alpha=0.8,marker=markerlist[jj])

        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.4)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_aspect('equal')
        if rownum==1:
            ax.set_title('$\Pr(Y='+str(ii)+'|X)$')
        if ii==0:
            ax.set_xlabel(classname)
        elif ii==1:
            ax.set_xlabel('Regret = ${:.4f}'.format(testRegret) + '$')

#######################################################################
def GetMNIST():
    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    X_train = np.array(train_dataset.data,dtype='float64')/255.
    if MNISTcrop:
        X_train = X_train[:,4:-4,4:-4].reshape((-1,20 * 20),order='C')
        ndim = 20 * 20
    else:
        X_train = X_train.reshape((-1,28 * 28),order='C')
        ndim = 28 * 28
    y_train = np.array(train_dataset.targets,dtype='int32')

    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())
    X_test = np.array(test_dataset.data,dtype='float64')/255.
    if MNISTcrop:
        X_test = X_test[:, 4:-4, 4:-4].reshape((-1, 20 * 20), order='C')
    else:
        X_test = X_test.reshape((-1, 28 * 28), order='C')
    y_test = np.array(test_dataset.targets,dtype='int32')
    return X_train, X_test, y_train, y_test, ndim

class GaussianModel:
    def __init__(self, K, ndim,noisegaindB, covweightdB, ToyExample=False):
        if ToyExample:
            self.K = 3
            self.ndim = 2
            self.centroids = np.zeros((K, ndim))
        else:
            self.K = K
            self.ndim = ndim

        self.centroids = np.zeros((self.K, self.ndim))
        self.covmats = np.zeros((self.K, self.ndim, self.ndim))
        self.invcovmats = np.zeros((self.K, self.ndim, self.ndim))
        self.sqrtcovmats = np.zeros((self.K, self.ndim, self.ndim))
        self.preexps = np.zeros(self.K)
        for ii in range(self.K):
            if ToyExample:
                if ii==0:
                    self.centroids[ii, :] = np.array([0, 3])
                    Amat = np.array([[1, 0], [0, 1]])
                elif ii==1:
                    self.centroids[ii, :] = np.array([-1, -1])
                    Amat = np.array([[1,0],[0,0.5]])
                elif ii==2:
                    self.centroids[ii, :] = np.array([+1, 0])
                    Amat = np.array([[0.5, 0], [0, 1]])
                covmat = Amat @ Amat.transpose()
            else:
                self.centroids[ii,:] = np.random.randn(ndim)
                # self.centroids[ii, :] =self.centroids[ii, :] * (10 ** (ii*0.5 / 10.))
                Amat = np.random.randn(ndim, ndim)
                # Amat = np.diag(np.random.randn(ndim))
                covright = Amat @ Amat.transpose()
                covright = covright / (LA.det(covright)**(1/ndim))
                # covright = covright * (10 ** (ii*1. / 10.))
                covmat =np.eye(ndim)+ 10**(covweightdB/10.)*covright
                covmat = covmat / (LA.det(covmat)**(1/ndim))
            covmat = covmat*10.**(noisegaindB/10.)
            self.covmats[ii, :, :] = covmat
            self.sqrtcovmats[ii, :, :] = sqrtm(covmat)
            self.invcovmats[ii, :, :] = LA.inv(covmat)
            if ToyExample:
                self.preexps[ii] = (2 * np.pi) ** (-ndim / 2) / np.sqrt(LA.det(covmat))
            else:
                self.preexps[ii] = (2 * np.pi) ** (-ndim / 2) / np.sqrt(LA.det(covmat))
                # self.preexps[ii] = np.array(1.)  # covmats are normalized so no issue

    def drawGaussianData(self,nsamps):
        y = np.random.randint(self.K, size=nsamps)
        X = np.zeros((nsamps, self.ndim))
        for ii in range(self.K):
            iinds = np.arange(nsamps)[y == ii]
            niinds = iinds.size
            X[y == ii, :] = np.tile(self.centroids[ii, :], (niinds, 1))
            addnoise = np.random.randn(niinds, self.ndim) @ self.sqrtcovmats[ii, :, :]
            X[y == ii, :] = X[y == ii, :] + addnoise
        return X, y

    def CalcGaussPDFval(self,X):
        nsamps,ndim = X.shape
        pdfmat = np.zeros((nsamps,self.K))
        for ii in range(self.K):
            Xcentered = X-np.tile(self.centroids[ii, :], (nsamps,1))
            exparg = -0.5*((Xcentered @ self.invcovmats[ii,:,:]) * Xcentered).sum(1)
            pdfval = np.zeros(exparg.shape)
            minexp = -300
            pdfval[exparg>minexp] = self.preexps[ii]  * np.exp(exparg[exparg>minexp])
            # pdfval = self.preexps[ii]  * np.exp(exparg)
            pdfmat[:,ii] = pdfval
        return pdfmat

    def calcpost(self,X):
        pdfmat = self.CalcGaussPDFval(X)
        normmat = np.tile(pdfmat.sum(1), (self.K, 1)).transpose()
        postmat = pdfmat / normmat
        return postmat

#######################################################################

def calcloglossX(X_pdfmat,y):
    nsamps = X_pdfmat.shape[0]
    probVals = X_pdfmat[np.arange(nsamps),y]
    logprobVals = -np.inf*np.ones(probVals.shape)
    logprobVals[probVals>0] = np.log(probVals[probVals>0])
    logloss = -logprobVals.mean()
    return logloss

#######################################################################

def calcloglossXlog(X_pdfmatlog, y):
    nsamps = X_pdfmatlog.shape[0]
    logprobVals = X_pdfmatlog[np.arange(nsamps), y]
    logloss = -logprobVals.mean()
    return logloss

#######################################################################

def myismember(a,b):
    return ismember(a,b)[0]

#######################################################################
#######################################################################

class classTreeNode:
    def __init__(self, cls, children, IsLeaf, Leverage=False, SGDparams={}):
        self.children = children
        self.IsLeaf = IsLeaf
        self.Leverage = Leverage
        self.SGDparams = SGDparams
        self.cls = cls

    def setleaves(self,leaves):
        self.allleaves  = flattenlist(leaves)
        self.zeroleaves = flattenlist(leaves[0])
        self.oneleaves  = flattenlist(leaves[1])

    def fit(self, X, y):
        if not (self.Leverage):
            self.fitbinary(X, y)
        else:
            self.fitmulti(X, y)

    def fitbinary(self,X,y):
        self.binclf = self.cls()
        nsamps = len(y)
        curlabels = self.allleaves
        onelabels = self.oneleaves
        curinds = np.arange(nsamps)[myismember(y, curlabels)]
        ycur = np.zeros(curinds.shape[0], dtype='int')
        ycur[myismember(y[curinds], onelabels)] = 1
        Xcur = X[curinds,]
        self.binclf.fit(Xcur, ycur)

    def maplabels(self,y):
        mapmat = np.tile(self.mapvec,(y.__len__(),1))
        return mapmat[range(y.__len__()), y]

    def calcWeightedSamples(self,XX,yy,betavecs,curlabel, allcurlabels):
        ndim = betavecs.shape[1]
        nK = len(allcurlabels)
        betamat = betavecs[allcurlabels,:] -np.tile(betavecs[curlabel,:],(nK,1))
        curinds = np.arange(yy.__len__())[myismember(yy, allcurlabels)]
        Xbeta = XX[curinds,:] @ betamat.T
        Wx = 1./np.exp(Xbeta).sum(1)
        Xweighted = (XX[curinds,:] * np.tile(Wx,(ndim,1)).T).sum(0)
        return Xweighted

    def calcGrad(self, Xcur, ycur,betavecs):
        alllabels  = self.maplabels(self.allleaves)
        zerolabels = self.maplabels(self.zeroleaves)
        onelabels  = self.maplabels(self.oneleaves)

        gradmat = np.zeros(betavecs.shape)
        for ii in range(alllabels.__len__()):
            gradleft = self.calcWeightedSamples(Xcur, ycur, betavecs, ii, alllabels)
            if myismember([ii],zerolabels)[0]:
                gradright = self.calcWeightedSamples(Xcur, ycur, betavecs, ii, zerolabels)
            else:
                gradright = self.calcWeightedSamples(Xcur, ycur, betavecs, ii, onelabels)
            gradmat[ii, :] = -(gradleft - gradright) #/ ycur.shape[0]
        return gradmat

    def fitmulti(self,X,y):
        X = addintercept(X)
        nsamps = len(y)
        curlabels = self.allleaves
        curinds = np.arange(nsamps)[myismember(y, self.allleaves)]
        ycur = y[curinds]
        Xcur = X[curinds,]

        # first, give numbers to all the remaining classes
        # self.mapvec = np.argsort(self.allleaves)
        self.mapvec = -1*np.ones(K,'int')
        for kk in range(len(self.allleaves)):
            self.mapvec[self.allleaves[kk]] = kk

        ynew = self.maplabels(ycur)
        if (len(curlabels) == 2):
            self.binclf = LogisticRegression(random_state=0, max_iter=1000,
                                             multi_class='auto', solver='lbfgs',fit_intercept=False)
            self.binclf.fit(Xcur,ynew)
            self.betavecs = np.r_[0*self.binclf.coef_,self.binclf.coef_]
            return # in the binary case this is already the classifier, no need for SGD
        else:
            curtime = timer()
            self.binclf = LogisticRegression(random_state=0, max_iter=1000,
                                             multi_class='multinomial', solver='lbfgs',fit_intercept=False)
            self.binclf.fit(Xcur,ynew)
            toc(curtime)
            self.betavecs = self.binclf.coef_
            # self.betavecs = np.random.randn(len(curlabels),ndim+1)
        # SGD
        ntrainsamps = ynew.__len__()
        nstepsperepoch = int(ntrainsamps / self.SGDparams['batch_size'])
        np.random.seed(SGDparams['sgdseed']) # set the randomness of the permutation

        sgdtime = timer()
        for ii in range(self.SGDparams['n_sgd_epochs']):
            indpermvec = np.random.permutation(ntrainsamps)
            for jj in range(nstepsperepoch):
                curinds = indpermvec[jj * SGDparams['batch_size']:min([(jj + 1) * SGDparams['batch_size'], ntrainsamps])]
                dVec = self.calcGrad(Xcur[curinds, :], ynew[curinds], self.betavecs)
                self.betavecs = self.betavecs + SGDparams['alpha'] * dVec
            if np.mod(ii+1,10)==0:
                print('finished SGD iteration #'+str(ii+1)+', ',end='')
                sgdtime = toc(sgdtime)
        return

    def predict_proba(self,X):
        if not (self.Leverage):
            proba = self.binclf.predict_proba(X)
        else:
            proba = np.zeros((X.shape[0],2))
            Xbeta = addintercept(X) @ self.betavecs.T
            expBeta = np.exp(Xbeta)
            zerolabels = self.maplabels(self.zeroleaves)
            proba[:, 0] = expBeta[:,zerolabels].sum(1)/expBeta.sum(1)
            proba[:, 1] = 1. - proba[:, 0]
        return proba

    def calcBinaryLoss(self,X,y):
        nsamps = len(y)
        curlabels = self.allleaves
        onelabels = self.oneleaves
        curinds = np.arange(nsamps)[myismember(y, curlabels)]
        ycur = np.zeros(curinds.shape[0], dtype='int')
        ycur[myismember(y[curinds], onelabels)] = 1
        Xcur = X[curinds,]
        curproba = self.predict_proba(Xcur)
        binloss = calcloglossX(curproba,ycur)
        # the weighing is according to the empirical probability
        nodeweight = curinds.__len__()/nsamps
        return binloss, nodeweight

#######################################################################

class HierarchiClass():
    def __init__(self,cls,K,children,isLeaf,Leverage,SGDparams={}):
        self.cls = cls
        self.K = K
        self.children = children
        self.isLeaf = isLeaf
        self.SGDparams = SGDparams
        self.Leverage = Leverage
        self.nodes = (K-1)*['']
        for ii in range(K-1):
            self.nodes[ii] = classTreeNode(self.cls, self.children[ii], self.isLeaf[ii],self.Leverage, self.SGDparams)
        for ii in range(K-1):
            self.nodes[ii].setleaves(self.findLeaves(ii))
            if set(self.nodes[ii].allleaves) == set(range(K)):
                self.rootnode = ii
        self.buildprefixcode()

    def findLeaves(self,ii):
        # find all the leaves related to a node
        leaves = [[],[]]
        for jj in range(2):
            curchild = self.nodes[ii].children[jj]
            if self.nodes[ii].IsLeaf[jj]:
                leaves[jj].append(curchild)
            else:
                leaves[jj].append(self.findLeaves(curchild))
        return leaves

    def buildprefixcode(self):
        self.ancestors = self.K * [[]]
        self.bitseqs   = self.K * [[]]
        for ii in range(self.K):
            curlabels = [ii]
            curancestors = []
            curbitseq = []
            rootreached = False
            while not(rootreached):
                ancestorfound = False
                jj = 0
                while not(ancestorfound):
                    if set(curlabels)==set(self.nodes[jj].zeroleaves):
                        curbitseq.append(0)
                        curancestors.append(jj)
                        curlabels = self.nodes[jj].allleaves
                        ancestorfound = True
                    elif set(curlabels) == set(self.nodes[jj].oneleaves):
                        curbitseq.append(1)
                        curancestors.append(jj)
                        curlabels = self.nodes[jj].allleaves
                        ancestorfound = True
                    else:
                        jj = jj+1
                rootreached = (curancestors[-1] == self.rootnode)
            self.ancestors[ii] = curancestors
            self.bitseqs[ii]   = curbitseq

    def fit(self, X, y):
        curtime = timer()
        for ii in range(K - 1):
            print('Training classifier #'+str(ii+1)+', ',end='')
            self.nodes[ii].fit(X,y)
            curtime = toc(curtime)

    def predict_proba(self,X):
        nsamps = X.shape[0]
        nodeprob = np.ones((nsamps,2,self.K-1))
        for jj in range(self.K-1):
            nodeprob[:,:,jj] = self.nodes[jj].predict_proba(X)

            # nodeprob[:,:,jj] = self.nodes[jj].binclf.predict_proba(X)
        classproba = np.ones((nsamps, self.K))
        for ii in range(self.K):
            for jj in range(len(self.ancestors[ii])):
                curnode = self.ancestors[ii][jj]
                curbit = self.bitseqs[ii][jj]
                classproba[:,ii] = classproba[:,ii]*nodeprob[:,curbit,curnode]
        return classproba

    def calcBinaryLosses(self, X, y, TrainTestStr):
        binarylossvec = np.zeros(K - 1)
        nodeweightvec = np.zeros(K - 1)
        totLogLoss = np.array(0)
        for ii in range(K - 1):
            binarylossvec[ii], nodeweightvec[ii] = self.nodes[ii].calcBinaryLoss(X,y)
            totLogLoss = totLogLoss + binarylossvec[ii]*nodeweightvec[ii]
        print(TrainTestStr+' loss vector = ['+formatArray(binarylossvec)+']')
        print('Total weighted loss = {:.6f} '.format(totLogLoss))
        return

#######################################################################

class OVAClass():
    def __init__(self,cls,K):
        self.cls = cls
        self.K = K

    def fit(self,X,y):
        nsamps = y.size
        self.OVAClassifiers = [self.cls]*self.K
        for ii in np.arange(self.K):
            ycur = 1*(y==ii)
            self.OVAClassifiers[ii] = self.cls()
            self.OVAClassifiers[ii].fit(X, ycur)

    def predict_proba(self,X):
        nsamps = X.shape[0]
        POVA = np.zeros((nsamps,self.K))
        for ii in np.arange(self.K):
            POVA[:,ii] = self.OVAClassifiers[ii].predict_proba(X)[:,1]
        POVA = POVA/np.tile(POVA.sum(1),(K,1)).T
        return POVA

    def predict_log_proba(self,X):
        nsamps = X.shape[0]
        POVA = np.zeros((nsamps,self.K))
        for ii in np.arange(self.K):
            POVA[:,ii] = self.OVAClassifiers[ii].predict_proba(X)[:,1]
        POVA = POVA/np.tile(POVA.sum(1),(K,1)).T
        return np.log(POVA)

    def CalcBinaryLosses(self,X,y,GaussModel=False):
        nsamps = X.shape[0]
        loglossvec = np.zeros(self.K)
        if GaussModel:
            Xpdfmat = GaussModel.CalcGaussPDFval(X)
        for ii in np.arange(self.K):
            ycur = 1*(y==ii)
            if not(GaussModel):
                curProbs = self.OVAClassifiers[ii].predict_proba(X)
            else:
                class0prob = Xpdfmat[:, ii]
                classrestprob = Xpdfmat.sum(1)
                class0prob = class0prob/classrestprob
                curProbs = np.c_[1-class0prob,class0prob]
            loglossvec[ii] = calcloglossX(curProbs,ycur)
        return loglossvec

#######################################################################

def TestModel(X_train, y_train,X_test,y_test,loglossRealtrain,loglossRealtest,curclfclass,classname):
    clfcur = curclfclass
    curtime = timer()
    clfcur.fit(X_train, y_train)
    clfcur.classname = classname
    Xtrain_post  = clfcur.predict_proba(X_train)
    y_train_hat = Xtrain_post.argmax(1)
    trainerr = (y_train_hat != y_train).mean()
    loglosstrain = calcloglossX(Xtrain_post,y_train)

    Xtest_post  = clfcur.predict_proba(X_test)
    y_test_hat = Xtest_post.argmax(1)
    testerr = (y_test_hat != y_test).mean()
    loglosstest = calcloglossX(Xtest_post,y_test)

    TrainRegret = loglosstrain-loglossRealtrain
    TestRegret = loglosstest-loglossRealtest
    print('\n' +classname+ '\n##################################')
    print('train loss = {:.6f}'.format(loglosstrain) +
          ', test loss  = {:.6f}'.format(loglosstest))

    if not(useMNIST):
        print('train Regret = {:.6f}'.format(TrainRegret) +
              ', test Regret  = {:.6f}'.format(TestRegret))
    else:
        print('train err = {:.6f}'.format(trainerr) +
              ', test err = {:.6f}'.format(testerr))

    return clfcur,loglosstrain,loglosstest,TrainRegret,TestRegret

#######################################################################

def TestRealModel(X_train,y_train,X_test,y_test):
    Xtrain_postReal = GaussModel.calcpost(X_train)
    loglossRealtrain = calcloglossX(Xtrain_postReal,y_train)
    Xtest_postReal = GaussModel.calcpost(X_test)
    loglossRealtest  = calcloglossX(Xtest_postReal,y_test)
    print('Optimal loglosses (Real Model)\n##################################')
    print('loglossRealtrain  = {:.6f}'.format(loglossRealtrain) +
          ', loglossRealtest = {:.6f}'.format(loglossRealtest))
    return loglossRealtrain,loglossRealtest

#######################################################################

def findinvec(v,cond): # a Matlab style find function
    return np.arange(v.shape[0])[cond]

#######################################################################
########## define the classifer objects ###############################

class myLogit(LogisticRegression):
    def __init__(self,LogitSGD=False,SGDparams={}):
        self.LogitSGD = LogitSGD
        self.SGDparams = SGDparams
        if not(useMNIST):
            LogisticRegression.__init__(self, random_state = 0,max_iter=100)#,multi_class='ovr') #'auto', 'ovr', 'multinomial'
        else:
            LogisticRegression.__init__(self, random_state=0, max_iter=1000, solver='lbfgs', multi_class='multinomial')#'multinomial') #‘newton-cg’

    def fit(self, X, y, sample_weight=None):
        if not(self.LogitSGD):
            LogisticRegression.fit(self,X, y, sample_weight)
        else:
            self.SGDfit(X,y)

    def predict_proba(self, X):
        if not(self.LogitSGD):
            return LogisticRegression.predict_proba(self,X)
        else:
            return self.predict_probaSGD(addintercept(X))

    def SGDfit(self, Xt, yt):
        Xt = addintercept(Xt)
        self.nclasses = yt.max()+1
        self.coef_ = np.random.randn(self.nclasses, Xt.shape[1])
        ntrainsamps = y_train.shape[0]
        nstepsperepoch = int(ntrainsamps / self.SGDparams['batch_size'])
        sgdtime = timer()
        for ii in range(self.SGDparams['n_sgd_epochs']):
            indpermvec = np.random.permutation(ntrainsamps)
            for jj in range(nstepsperepoch):
                curinds = indpermvec[jj * self.SGDparams['batch_size']:min([(jj + 1) * self.SGDparams['batch_size'], ntrainsamps])]
                dVec = self.calamylogitGrad(Xt[curinds, :], yt[curinds])
                self.coef_ = self.coef_ + self.SGDparams['alpha'] * dVec
            if np.mod(ii + 1, 10) == 0:
                print('finished SGD iteration #' + str(ii + 1) + ', ', end='')
                sgdtime = toc(sgdtime)
        return

    def calamylogitGrad(self, Xcur, ycur):
        Xbeta = Xcur @ self.coef_.T
        gradmat = np.zeros(self.coef_.shape)

        for ii in range(self.nclasses):
            gradleft = Xcur[ycur == ii].sum(0)
            Xbetacur = Xbeta - np.tile(Xbeta[:, ii], (Xbeta.shape[1], 1)).T
            dcoeff = 1 / np.tile(np.exp(Xbetacur).sum(1), (Xcur.shape[1], 1)).T
            gradright = (Xcur * dcoeff).sum(0)
            gradmat[ii, :] = (gradleft - gradright) #/ ycur.shape[0]
        return gradmat

    def predict_probaSGD(self,Xcur):
        Xbeta = Xcur @ self.coef_.T
        expBeta = np.exp(Xbeta)
        denom = np.tile(expBeta.sum(1), (self.nclasses, 1)).T
        return expBeta / denom

#######################################################################
class myRF(RandomForestClassifier):
    def __init__(self):
        RandomForestClassifier.__init__(self,max_depth=5, random_state=0)

#######################################################################

def predicandlogloss(Xcur,ycur,cls):
    logpostprobcur = cls.predict_log_proba(Xcur)
    return  calcloglossXlog(logpostprobcur, ycur)

#######################################################################

def predicandloglossBin(Xcur,ycur,cls):
    postprobcur = cls.predict_proba(Xcur)
    llvec = np.zeros(2)
    llvec[0] = calcloglossX(postprobcur, ycur)
    postprobbin = np.zeros((postprobcur.shape[0],2))
    postprobbin[:,0] = postprobcur[:,0]
    postprobbin[:,1] = postprobcur[:,1:].sum(1)
    llvec[1] = calcloglossX(postprobbin, 1*(ycur>0))
    return llvec

#######################################################################

def predict_probmy(Xcur,curbeta):
    Xbeta = Xcur @ curbeta.T
    expBeta = np.exp(Xbeta)
    denom = np.tile(expBeta.sum(1),(2,1)).T
    return expBeta/denom

#######################################################################

def calclossMB(Xcur,ycur,curbeta):
    K = curbeta.shape[0]
    Xbeta = Xcur @ curbeta.T
    expBeta = np.exp(Xbeta)
    denom = np.tile(expBeta.sum(1),(K,1)).T
    postprobcur = expBeta/denom
    postprobbin = np.zeros((postprobcur.shape[0],2))
    postprobbin[:,0] = postprobcur[:,0]
    postprobbin[:,1] = postprobcur[:,1:].sum(1)
    ll = calcloglossX(postprobbin, 1*(ycur>0))
    return ll,postprobbin

#######################################################################

def calcLossMy(Xcur,ycur,curbeta):
    return calcloglossX(predict_probmy(Xcur,curbeta), ycur)

#######################################################################

addintercept = lambda XX: np.c_[np.ones((XX.shape[0], 1)), XX]

#######################################################################

class LogitBinMulti():
    def __init__(self,K,SGDparams):
        self.K = K
        self.SGDparams = SGDparams

    def calcMBlogitGrad(self,Xcur, ycur):
        gradmat = np.zeros(self.betavecs.shape)
        Xbeta = Xcur @ self.betavecs.T
        # swap labels of oneclass and 0

        # calculate beta0
        gradleft = Xcur[ycur == 0, :].sum(0)
        Xbetacur = Xbeta - np.tile(Xbeta[:, 0], (Xbeta.shape[1], 1)).T
        dcoeff = 1 / np.tile(np.exp(Xbetacur).sum(1), (Xcur.shape[1], 1)).T
        gradright = (Xcur * dcoeff).sum(0)
        gradmat[0, :] = (gradleft - gradright) / ycur.shape[0]
        # calculate the other betas
        for ii in range(1, self.K):
            Xcurii = Xcur[ycur >0 , :]
            Xbetaii = Xbeta[ycur >0, 1:] - np.tile(Xbeta[ycur>0, ii], (Xbeta.shape[1] - 1, 1)).T
            dcoeffii = 1 / np.tile(np.exp(Xbetaii).sum(1), (Xcurii.shape[1], 1)).T
            # dcoeffii = np.array(1.)
            gradleft = (Xcurii * dcoeffii).sum(0)
            Xbetacur = Xbeta - np.tile(Xbeta[:, ii], (Xbeta.shape[1], 1)).T
            dcoeff = 1 / np.tile(np.exp(Xbetacur).sum(1), (Xcur.shape[1], 1)).T
            gradright = (Xcur * dcoeff).sum(0)
            gradmat[ii, :] = (gradleft - gradright) / ycur.shape[0]
        return gradmat

    def fit(self,Xt,yt, oneclass):
        Xt = addintercept(Xt)
        # swap labels of oneclass and zero
        oneclassinds = np.arange(yt.shape[0])[yt==oneclass]
        zeroinds = np.arange(yt.shape[0])[yt == 0]
        yt[oneclassinds] = 0
        yt[zeroinds] = oneclass
        ntrainsamps,ndim = Xt.shape

        if self.K==2:
            multilogitcls = LogisticRegression(random_state=0, max_iter=100,
                                               multi_class='auto', fit_intercept=False)
            # setting multi_class = 'auto' in the binary case is important
            # setting 'multinomial' yields the wrong coeffs here
            multilogitcls.fit(Xt, yt)
            self.betavecs =    np.r_[0*multilogitcls.coef_,multilogitcls.coef_]
            return
        # if self.K>2
        multilogitcls = LogisticRegression(random_state=0, max_iter=100,
                                           multi_class='multinomial',fit_intercept=False)
        multilogitcls.fit(Xt, yt)
        self.betavecs = multilogitcls.coef_
        nstepsperepoch = int(ntrainsamps / self.SGDparams['batch_size'])
        np.random.seed(SGDparams['sgdseed']) # set the randomness of the permutation

        for ii in range(self.SGDparams['n_sgd_epochs']):
            indpermvec = np.random.permutation(ntrainsamps)
            for jj in range(nstepsperepoch):
                curinds = indpermvec[jj * SGDparams['batch_size']:min([(jj + 1) * SGDparams['batch_size'], ntrainsamps])]
                dVec = self.calcMBlogitGrad(Xt[curinds, :], yt[curinds])
                self.betavecs = self.betavecs + self.SGDparams['alpha'] * dVec

    def predict_proba(self,Xcur):
        Xcur = addintercept(Xcur)
        Xbeta = Xcur @ self.betavecs.T
        expBeta = np.exp(Xbeta)
        denom = np.tile(expBeta.sum(1),(self.K,1)).T
        postprobcur = expBeta/denom
        postprobbin = np.zeros((postprobcur.shape[0],2))
        postprobbin[:,1] = postprobcur[:,0]
        postprobbin[:,0] = postprobcur[:,1:].sum(1)
        return postprobbin

    def predict_log_proba(self,Xcur):
        return np.log(self.predict_proba(Xcur))

########################################################################################################################
##########   MAIN    ###################################################################################################
########################################################################################################################
curtime = timer()
starttime = curtime
clfClass = myLogit
#######################################################################
# settings
useMNIST = False
syntheticSenarioAorB = True
nsamps = int(10**6*10/6) # number of samples for the synthetic experiemnt
#######################################################################

# configuration for the MNIST case
MNISTpermute = False
MNISTcrop = True

# configuration for the synthetic case
ToyExample = False
randseed = 10
K = 10
ndim = 100
noisegaindB = np.array(20.)

# SGD configuration
SGDparams ={}
SGDparams['n_sgd_epochs'] = 100
SGDparams['batch_size'] = 1000
SGDparams['sgdseed'] = 1

if syntheticSenarioAorB:
    covweightdB = -50.
else:
    covweightdB = -10.

if useMNIST:
    SGDparams['alpha'] = 0.001
    BalancedorCOVAtree = True
else:
    SGDparams['alpha'] = 0.000001
    BalancedorCOVAtree = False


#######################################################################
######## heirarchical classification tree

if ToyExample:
    childrenvec = [[0,1],[1,2]]
    isLeafvec = [[True,False],[True,True]]
else:
    if BalancedorCOVAtree:
        childrenvec = [[0, 1], [2, 3], [0, 1], [4, 5], [6, 7], [8, 9], [2, 3], [4, 5], [6, 7]]
        isLeafvec = [[True, True], [True, True], [False, False], [True, True], [True, True], [True, True],
                     [False, False], [False, False], [False, False]]
    else:
        childrenvec = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9]]
        isLeafvec = [[True,False],[True,False],[True,False],[True,False],[True,False],[True,False],[True,False],[True,False],[True,True]]

#######################################################################

np.seterr(all='raise')
np.random.seed(randseed)
np.seterr(under='ignore')

######## choose classifier
if useMNIST:
    print('Using MNIST')
    X_train, X_test, y_train, y_test, ndim = GetMNIST()
    K = 10
    loglossRealtrain, loglossRealtest = np.array([0.,0.]) # no reference in the MNIST case
    if MNISTpermute:
        inpermvec = np.random.permutation(K)
        print('label permutation = '+str(inpermvec))
        perminds = lambda permvec, y: np.tile(inpermvec, (y.__len__(), 1))[range(y.__len__()), y]
        y_train = perminds(inpermvec,y_train)
        y_test = perminds(inpermvec,y_test)
else:
    if ToyExample:  # override the settings
        K = 3
        ndim = 2
        noisegaindB = 10.  # noise gain in dB
        covweightdB = 0.  # irrelevalnt in this case

    print('ToyExample = ' + str(ToyExample) + ', seed = ' + str(randseed) + ', nsamps = ' + str(nsamps))
    print('K = ' + str(K) + ', ndim = ' + str(ndim) + ', n_sgd_epochs = ' + str(SGDparams['n_sgd_epochs']))
    print('covweightdB = ' + str(covweightdB) + ', noisegaindB = ' + str(noisegaindB))

    ######## set the model parameters
    GaussModel = GaussianModel(K, ndim, noisegaindB, covweightdB, ToyExample)

    #######################################################################

    # draw the data
    X, y = GaussModel.drawGaussianData(nsamps)

    #######################################################################

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    # calculate real probabilities and log-losses
    print('Starting with real model')
    loglossRealtrain, loglossRealtest = TestRealModel(X_train, y_train, X_test, y_test)

########################################################################################################################
#################### Classifiers #######################################################################################
########################################################################################################################
#  multi-class classifier
if True:
    print('\nstarting Multiclass')
    clfM,loglossMtrain,loglossMtest,TrainRegretM,TestRegretM = \
        TestModel(X_train, y_train,X_test, y_test,loglossRealtrain,loglossRealtest,clfClass(),'Multiclass')
    curtime = toc(curtime)

if False:
    print('\nstarting Logit SGD')
    clfM,loglossMtrain,loglossMtest,TrainRegretM,TestRegretM = \
        TestModel(X_train, y_train,X_test, y_test,loglossRealtrain,loglossRealtest,myLogit(True, SGDparams),'Logit SGD')
    curtime = toc(curtime)

if True:
    # OVA classifier
    print('\nstarting OVA')
    OVAModel,loglossOVAtrain,loglossOVAtest,TrainRegretOVA,TestRegretOVA =\
        TestModel(X_train, y_train,X_test, y_test,loglossRealtrain,loglossRealtest,OVAClass(clfClass,K),'OVA')
    curtime = toc(curtime)

if True:
    print('\nstarting Heirarchical - no Leverage')
    Hclass,loglossHCtrain,loglossHCtest,TrainRegretHC,TestRegretHC =\
        TestModel(X_train, y_train,X_test, y_test,loglossRealtrain,loglossRealtest,\
                  HierarchiClass(clfClass,K,childrenvec,isLeafvec,False,SGDparams),'Hclass')
    binaryLossTrain = Hclass.calcBinaryLosses(X_train, y_train, 'train')
    binaryLossTest = Hclass.calcBinaryLosses(X_test, y_test, 'test')
    curtime = toc(curtime)

if True:
    print('\nstarting Heirarchical - with Leverage')
    Hclass,loglossHCtrain,loglossHCtest,TrainRegretHC,TestRegretHC =\
        TestModel(X_train, y_train,X_test, y_test,loglossRealtrain,loglossRealtest,\
                  HierarchiClass(clfClass,K,childrenvec,isLeafvec,True,SGDparams),'L-Hclass')
    binaryLossTrain = Hclass.calcBinaryLosses(X_train, y_train, 'train')
    binaryLossTest = Hclass.calcBinaryLosses(X_test, y_test, 'test')
    curtime = toc(curtime)

##########################################################################