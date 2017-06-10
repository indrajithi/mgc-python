# File: svm.py
# Author: Indrajith Indraptrastham
# Date: Sun Apr 30 23:23:52 IST 2017


import numpy as np 
<<<<<<< HEAD
from . import acc
from . import feature
=======
import acc
import feature
>>>>>>> ce120f1d2dfdc28237def03235925ff37c879e8b
from sklearn import svm
from sklearn.model_selection import cross_val_score
import random
import itertools
<<<<<<< HEAD
import os
import pkg_resources
from sklearn.externals import joblib
#load pre saved variables
resource_package = __name__
resource_path = '/'.join(('', 'data/Xall.npy'))

Xall_path = pkg_resources.resource_filename(resource_package, resource_path)
Xall = np.load(Xall_path)
Yall = feature.geny(100)

resource_path = '/'.join(('', 'data/classifier_10class.pkl'))
clf_path = pkg_resources.resource_filename(resource_package, resource_path)
myclf = joblib.load(clf_path)

#classical metal pop rock
resource_path = '/'.join(('', 'data/cmpr.pkl'))
clf_path = pkg_resources.resource_filename(resource_package, resource_path)
cmpr = joblib.load(clf_path)

resource_path = '/'.join(('', 'data/classifier_10class_prob.pkl'))
clf_path = pkg_resources.resource_filename(resource_package, resource_path)
clfprob = joblib.load(clf_path)

label = feature.getlabels()


def poly(X,Y):
    """
    Returns a polynomial kernal svm
    Args:
    """
    #Polynomial kernel=======================================================
    clf = svm.SVC(kernel='poly',C=1,probability=True)
    clf.fit(X,Y)

    return clf


def fit(train_percentage,fold=5):
    """ Radomly choose songs from the dataset, and train the classfier 
        Accepts parameter: train_percentage, fold;
        Returns clf    
    """
    
    resTrain =0
    resTest = 0
    score = 0
    scores = 0

    for folds in range(fold):
        #init
        flag = True
        flag_train = True
        start = 0
        train_matrix = np.array([])
        test_matrix = np.array([])
        Xindex = []
        Tindex = []

        for class_counter in range(10):
            stack = list(range(start, start+100))  #create an index of size 100
            for song_counter in range( int(train_percentage) ):
                index = random.choice(stack)      #randomly choose numbers from index
                stack.remove(index)               #remove the choosen number from index
                random_song = Xall[index]         #select songs from that index for training
                Xindex.append(index)
                if flag:
                    train_matrix = random_song
                    flag = False
                else:
                    train_matrix = np.vstack([train_matrix, random_song])
            start += 100

            #select the remaning songs from the stack for testing
            for test_counter in range(100 - train_percentage):
                Tindex.append(stack[test_counter])
                if flag_train:
                    test_matrix = Xall[stack[test_counter]]
                    flag_train = False
                else:
                    test_matrix = np.vstack([test_matrix, Xall[stack[test_counter]]])

        Y = feature.geny(train_percentage) 
        y = feature.geny(100 - train_percentage)

        clf = svm.SVC(kernel='poly',C=1,probability=True)
        clf.fit(train_matrix, Y)
        #training accuracy
        res = clf.predict(test_matrix)
        ac = acc.get(res,y)
        print("accuracy = ", ac)
        return ac, clf


def getprob(filename):
    """
    Find the probality that a song belongs to each genre. 
    """
    x = feature.extract(filename)
    clf = cmpr
    prob = clf.predict_proba(x)[0]
    #prob = np.round(prob,decimals=-5)
    #dd = dict(zip(feature.getlabels(),prob))
    dd = dict(zip(['Classical','Hipop','Jass','Metal','Pop','Rock'],prob))
    print(prob)

    # max probablity 
    m = max(dd,key=dd.get)
    print(m, dd[m])

    sorted_genre = sorted(dd,key=dd.get,reverse=True)
    has_features_of = []
    for i in sorted_genre:
        if (dd[i] > 0.15 or dd[i] >= dd[m]) and len(has_features_of) < 3:
            has_features_of.append(i)


    return dd, has_features_of
    



def random_cross_validation(train_percentage,fold):
    """ 
    Randomly crossvalidate with training percentage and fold. Accepts parameter: train_percentage, fold;
    """

=======

#load pre saved variables
X90 = np.load('npy/X90.npy')
Y90 = feature.geny(90)
xt10 = np.load('npy/X10.npy')
yt10 = feature.geny(10)
Xall = np.load('npy/Xall.npy')
Yall = feature.geny(100)
"""
def svms():
    #linear kernel=======================================================
    print('='*5+' Linear Kernel SVM '+'='*5)
    clf = svm.LinearSVC(C=10000)
    clf.fit(X90,Y90)

    #cross validation
    resub = clf.predict(X90)
    resubAcc = acc.get(resub,Y90)
    #test case
    print('Resubstituion Accuracy : ',resubAcc)
    resTest = clf.predict(xt10)
    tesAcc = acc.get(resTest,yt10)
    print('Test case accuracy : ',tesAcc)
    scores = cross_val_score(clf, X90, Y90, cv=5)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('='*30)

    #Polynomial kernel=======================================================
    print('='*5+' Polynomial Kernel SVM '+'='*5)
    clf = svm.SVC(kernel='poly',C=1000)
    clf.fit(X,Y)

    #cross validation
    resub = clf.predict(X)
    resubAcc = acc.get(resub,Y)
    #test case
    print('Resubstituion Accuracy : ',resubAcc)
    resTest = clf.predict(xt10)
    tesAcc = acc.get(resTest,yt10)
    print('Test case accuracy : ',tesAcc)

    scores = cross_val_score(clf, xt, yt, cv=5)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('='*30)

"""

def poly(X,Y):
    #Polynomial kernel=======================================================
    clf = svm.SVC(kernel='poly',C=1)
    clf.fit(X,Y)
    #cross validation
    #resub = clf.predict(X)
    #resubAcc = acc.get(resub,Y)
    #test case
    #resTest = clf.predict(xt)
    #tesAcc = acc.get(resTest,yt)
    #scores = cross_val_score(clf, X, Y, cv=5)

    #return scores.mean()
    return clf
    
     

def cross_validation(train_percentage,fold):
    """ Accepts parameter: 
            train_percentage;
            fold;
    """
    #creates a matrix of size 10x100x104
>>>>>>> ce120f1d2dfdc28237def03235925ff37c879e8b
    resTrain =0
    resTest = 0
    score = 0
    scores = 0

    for folds in range(fold):
        #init
        flag = True
        flag_train = True
        start = 0
        train_matrix = np.array([])
        test_matrix = np.array([])
        Xindex = []
        Tindex = []

        for class_counter in range(10):
            stack = list(range(start, start+100))  #create an index of size 100
            for song_counter in range( int(train_percentage) ):
                index = random.choice(stack)      #randomly choose numbers from index
                stack.remove(index)               #remove the choosen number from index
                random_song = Xall[index]         #select songs from that index for training
                Xindex.append(index)
                if flag:
                    train_matrix = random_song
                    flag = False
                else:
                    train_matrix = np.vstack([train_matrix, random_song])
            start += 100

            #select the remaning songs from the stack for testing
            for test_counter in range(100 - train_percentage):
                Tindex.append(stack[test_counter])
                if flag_train:
                    test_matrix = Xall[stack[test_counter]]
                    flag_train = False
                else:
                    test_matrix = np.vstack([test_matrix, Xall[stack[test_counter]]])

<<<<<<< HEAD
        Y = feature.geny(train_percentage) 
        y = feature.geny(100 - train_percentage)
        #training accuracy
        clf = svm.SVC(kernel='poly',C=1,probability=True)
        clf.fit(train_matrix, Y)
 
=======
        #print(train_matrix.shape)
        #print(test_matrix.shape)

        Y = feature.geny(train_percentage) 
        y = feature.geny(100 - train_percentage)
        #training accuracy
        clf = svm.SVC(kernel='poly',C=1)
        clf.fit(train_matrix, Y)
        #train case
        #scores = cross_val_score(clf, train_matrix, feature.geny(train_percentage), cv=5)
        #print("acc train",scores.mean())
        #test case
        #scores = cross_val_score(clf, test_matrix, feature.geny(100 - train_percentage), cv=5)
        #print("acc test",scores.mean())
>>>>>>> ce120f1d2dfdc28237def03235925ff37c879e8b
        res = clf.predict(train_matrix)
        #print(acc.get(res,Y))
        resTrain += acc.get(res,Y)
        res = clf.predict(test_matrix)
        resTest += acc.get(res,y)
<<<<<<< HEAD
    
    print("Training accuracy with %d fold %f: " % (int(fold), resTrain / int(fold)))
    print("Testing accuracy with %d fold %f: " % (int(fold), resTest / int(fold)))
    

=======
        #print(acc.get(res,y))
        #return test_matrix, train_matrix
        #scores = cross_val_score(clf, test_matrix, y, cv=5)
        #print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
        #score = 100 *clf.score(test_matrix,y)
        #print(score)
        #scores += score

    #print(scores/fold)
    #print(resTrain)
    #print(resTest)
    print("Training accuracy with %d fold %f: " % (int(fold), resTrain / int(fold)))
    print("Testing accuracy with %d fold %f: " % (int(fold), resTest / int(fold)))
>>>>>>> ce120f1d2dfdc28237def03235925ff37c879e8b


def findsubclass(class_count):
    """ 
<<<<<<< HEAD
    Returns all possible ways we can combine the classes. 
    Accepts an integer as class count
=======
    Finds the combination of classes NCR
>>>>>>> ce120f1d2dfdc28237def03235925ff37c879e8b
    """
    class_l = list(range(10))
    flag = True
    labels = np.array([])
    for i in itertools.combinations(class_l,class_count):
        if flag:
            labels = i
            flag = False
        else:
            labels = np.vstack([labels, i])
    return labels

<<<<<<< HEAD



def gen_sub_data(class_l):
    """
    Generate a subset of the dataset for the given list of classes
    """
=======
def gen_sub_data(class_l):
>>>>>>> ce120f1d2dfdc28237def03235925ff37c879e8b
    all_x = np.array([])
    flag = True;

    for class_index in class_l:
        if class_index != 0:
            class_index *= 100
        if flag:
            all_x = Xall[ class_index : class_index + 100 ]
            flag = False
        else: 
            all_x = np.vstack([all_x, Xall[ class_index : class_index + 100 ]])
    
    return all_x


def fitsvm(Xall,Yall,class_l,train_percentage,fold):
<<<<<<< HEAD
    """ 
    Fits an poly svm and returns the accuracy
    Accepts parameter: 
            train_percentage;
            fold;
    Returns: classifier, Accuracy
    """
=======
    """ Accepts parameter: 
            train_percentage;
            fold;
    """
    #creates a matrix of size 10x100x104
>>>>>>> ce120f1d2dfdc28237def03235925ff37c879e8b
    resTrain =0
    resTest = 0
    score = 0
    scores = 0

    for folds in range(fold):
        #init
        flag = True
        flag_train = True
        start = 0
        train_matrix = np.array([])
        test_matrix = np.array([])
        Xindex = []
        Tindex = []

        for class_counter in range(class_l):
            stack = list(range(start, start+100))  #create an index of size 100
            for song_counter in range( int(train_percentage) ):
                index = random.choice(stack)      #randomly choose numbers from index
                stack.remove(index)               #remove the choosen number from index
                random_song = Xall[index]         #select songs from that index for training
                Xindex.append(index)
                if flag:
                    train_matrix = random_song
                    flag = False
                else:
                    train_matrix = np.vstack([train_matrix, random_song])
            start += 100

            #select the remaning songs from the stack for testing
            for test_counter in range(100 - train_percentage):
                Tindex.append(stack[test_counter])
                if flag_train:
                    test_matrix = Xall[stack[test_counter]]
                    flag_train = False
                else:
                    test_matrix = np.vstack([test_matrix, Xall[stack[test_counter]]])
        Y = feature.gen_suby(class_l, train_percentage) 
        y = feature.gen_suby(class_l, 100 - train_percentage)
        #training accuracy
<<<<<<< HEAD
        clf = svm.SVC(kernel='poly',C=1,probability=True)
=======
        clf = svm.SVC(kernel='poly',C=1)
>>>>>>> ce120f1d2dfdc28237def03235925ff37c879e8b
        clf.fit(train_matrix, Y)
        #train case
        res = clf.predict(train_matrix)
        #print(acc.get(res,Y))
        resTrain += acc.get(res,Y)
        res = clf.predict(test_matrix)
        resTest += acc.get(res,y)

<<<<<<< HEAD
    return clf , resTest / int(fold)

def best_combinations(class_l, train_percentage, fold):
    """
    Finds all possible combination of classes and the accuracy for the given number of classes
    Accepts: Training percentage, and number of folds
    Returns: A List of best combination possible for given the class count.
    """
=======
    return resTest / int(fold)

def best_combinations(class_l, train_percentage, fold):
>>>>>>> ce120f1d2dfdc28237def03235925ff37c879e8b
    class_comb = findsubclass(class_l)
    avg = []
    X = gen_sub_data(class_comb[0])
    Y = feature.gen_suby(class_l,100)
    for class_count in range(class_comb.shape[0]):
        all_x = gen_sub_data( class_comb[ class_count ] )
        all_y = feature.gen_suby(class_l,100)
<<<<<<< HEAD
        clf , score = fitsvm(all_x, all_y, class_l, train_percentage, fold)
        avg.append(score)
        print(score)
        print(class_count)
    maxAvg = max(avg)
    maxIndex = [i for i, j in enumerate(avg) if j >= (maxAvg - 2)]
    print("Maximum accuracy:",maxAvg)
    print("Best combinations:")
    for i in maxIndex:
        print(class_comb[i])
    return  avg


def getGenre(filename):

    music_feature =  feature.extract(os.path.abspath(os.path.dirname(__name__)) \
        +'/django-jquery-file-upload/' +filename)
    clf = cmpr
    return clf.predict(music_feature)

def getMultiGenre(filename):

    #music_feature =  feature.extract(os.path.abspath(os.path.dirname(__name__)) \
    #   +'/django-jquery-file-upload/' +filename)
    
    dd, has_features_of = getprob(os.path.abspath(os.path.dirname(__name__)) \
        +'/django-jquery-file-upload/' +filename)

    return dd, has_features_of 
=======
        score = fitsvm(all_x, all_y, class_l, train_percentage, fold)
        avg.append(score)
        print(score)
        print(class_count)
    return avg





>>>>>>> ce120f1d2dfdc28237def03235925ff37c879e8b
