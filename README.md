Python package *mysvm*
======================

We developed a python package called `mysvm` which contains three modules: *features, svm, acc*. These are used by the web application in feature extraction and finding genre. This package also contains many other functions to do complicated feature extraction and classification.
```
├── acc.py
├── data
│   ├── classifier_10class.pkl
│   ├── classifier_10class_prob.pkl
│   ├── cmpr.pkl
│   └── Xall.npy
├── feature.py
├── __init__.py
├── svm.py
```
### *feature* 
This module is used to extract MFCC features from a given file. It contains the following functions.
* ***extract (file):*** 
Extract features from a given file. Files in other formats are converted to .wav format. Returns numpy array.
* ***extract_all (audio_dir):*** 
Extracts features from all files in a directory. 
* ***extract_ratio (train_ratio, test_ratio, audio_dir) :*** 
Extract features from all files in a directory in a ratio. Returns two numpy arrays.
* ***geny(n):*** 
Generates `Y` values for `n` classes. Returns numpy array.
* ***gen_suby(sub, n):***
Generates `Y` values for a subset of classes. Returns numpy array.
* ***gen_labels( ):***
Returns a list of all genre labels.
* ***flattern( x) :***
Flatterns a numpy array.
 
### *svm* 
A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. This module contains various functions for classification using support vector machines.
* ***poly(X,Y):*** 
Trains a poly kernel SVM by fitting X, Y dataset. Returns a trained poly kernel SVM classifier.
* ***fit ( training_percentage, fold):*** 
Randomly choose songs from the dataset, and train the classifier. Accepts parameter: `train_percentage, fold`; Returns trained classifier.
* ***getprob (filename):***
Find the probabilities for a song belongs to each genre. Returns a dictionary mapping genre names to probability and a list of top 3 genres which is having probability of more than 0.15. 
* ***random_cross_validation (train_percentage,fold):***
Randomly cross validate with training percentage and fold. Accepts parameter: `train_percentage, fold`;
* ***findsubclass (class_count):***
Returns all possible ways we can combine the classes. Accepts an integer as class count. Returns numpy array of all possible combination.
* ***gen_sub_data (class_l):***
Generate a subset of the dataset for the given list of classes. Returns numpy array.
* ***fitsvm (Xall, Yall, class_l, train_percentage, fold):***
Fits a poly kernel svm and returns the accuracy. Accepts parameter: `train_percentage; fold`;  Returns: classifier, Accuracy.

* ***best_combinations (class_l, train_percentage, fold):***
Finds all possible combination of classes and the accuracy for the given number of classes Accepts: Training percentage, and number of folds Returns: A List of best combination possible for given the class count.

* ***getgenre (filename):***
Accepts a filename and returns a genre label for a given file.

* ***getgenreMulti (filename):***
Accepts a filename and returns top three genre labels based on the probability. 

### *acc*
 Module for finding the accuracy.
* ***get ( res, test ) :***  
Compares two arrays and returns the accuracy of their match.

Results
=======

| Classifier | Training Accuracy | Testing Accuracy |
|:---------: | :---------------: | :--------------: |
| K-Nearest Neighbors        |                   | 53%              |
| Logistic Regression|  75.778% | 54% |
| SVM Linear Kernel | 99% | 52% |
| SVM RBF Kernel | 99% | 12% |
| **SVM Poly Kernel** | **99%** | **64%** |

**While choosing 6 genre classes we are getting an accuracy of 85%**
