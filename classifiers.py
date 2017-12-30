import numpy as np
from random import *

from sklearn import datasets
from numpy import genfromtxt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


###### List of Classifier 
from sklearn import svm
from sklearn import tree
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier


import Utils

PCA_COMPONENT_NO = 13
NO_OF_SPLITS = [3, 5, 10]
SEED = randint(1, 9999999)
'''
classifier_dictionary = [
    ["LR_C0", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1)) , "logistic regression C=1"],
    ["LR_C1", LogisticRegression(C=1e1), "logistic regression C=1e1"],
    ["LR_C2", LogisticRegression(C=1e2), "logistic regression C=1e2"],
    ["LR_C3", LogisticRegression(C=1e3), "logistic regression C=1e3"],
    ["LR_C4", LogisticRegression(C=1e4), "logistic regression C=1e4"],
    ["LR_C5", LogisticRegression(C=1e5), "logistic regression C=1e5"],
    ["LR_C6", LogisticRegression(C=1e6), "logistic regression C=1e6"],
    ["RF20", RandomForestClassifier(n_estimators=20), "Random Forest 20 estimator"],
    ["RF10", RandomForestClassifier(n_estimators=10), "Random Forest 10 estimator"],
    ["RF5", RandomForestClassifier(n_estimators=5), "Random Forest 5 estimator"],
    ["RF2", RandomForestClassifier(n_estimators=2), "Random Forest 2 estimator"],
    ["DT", tree.DecisionTreeClassifier(), "Random Forest 10 estimator"],
    ["SVM_RBF", svm.SVC(kernel='rbf', C = 1), "svm.SVC kernel=linear"],
    ["SVM_LINEAR", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), svm.SVC(kernel='linear', C = 1)), "svm.SVC kernel=linear"],
    ["SVM_SIGMOID", svm.SVC(kernel='sigmoid', C = 1), "svm.SVC kernel=linear"],
    ["KNN1", KNeighborsClassifier(n_neighbors=1), "KNN k = 1"],
    ["KNN2", KNeighborsClassifier(n_neighbors=2), "KNN k = 2"],
    ["KNN5", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=5)), "KNN k = 5"],
    ["KNN10", KNeighborsClassifier(n_neighbors=10), "KNN k = 10"],
    ["KNN20", KNeighborsClassifier(n_neighbors=20), "KNN k = 20"],
    ["KNN50", KNeighborsClassifier(n_neighbors=50), "KNN k = 50"] 
    
]
'''


classifier_dictionary = [
    ["LR_C0", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1)) , "logistic regression C=1"],
    ["LR_C1", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e1)), "logistic regression C=1e1"],
    ["LR_C2", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e2)), "logistic regression C=1e2"],
    ["LR_C3", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e3)), "logistic regression C=1e3"],
    ["LR_C4",  make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e4)), "logistic regression C=1e4"],
    ["LR_C5", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e5)), "logistic regression C=1e5"],
    ["LR_C6", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), LogisticRegression(C=1e6)), "logistic regression C=1e6"],
    ["RF20", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), RandomForestClassifier(n_estimators=20)), "Random Forest 20 estimator"],
    ["RF10", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), RandomForestClassifier(n_estimators=10)), "Random Forest 10 estimator"],
    ["RF5", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), RandomForestClassifier(n_estimators=5)), "Random Forest 5 estimator"],
    ["RF2", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), RandomForestClassifier(n_estimators=2)), "Random Forest 2 estimator"],
    ["DT", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), tree.DecisionTreeClassifier()), "Random Forest 10 estimator"],
    ["SVM_RBF", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), svm.SVC(kernel='rbf', C = 1)), "svm.SVC kernel=linear"],
    ["SVM_LINEAR", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), svm.SVC(kernel='linear', C = 1)), "svm.SVC kernel=linear"],
    ["SVM_SIGMOID", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), svm.SVC(kernel='sigmoid', C = 1)), "svm.SVC kernel=linear"],
    ["KNN1", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=1)), "KNN k = 1"],
    ["KNN2", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=2)), "KNN k = 2"],
    ["KNN5", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=5)), "KNN k = 5"],
    ["KNN10", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=10)), "KNN k = 10"],
    ["KNN20", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=20)), "KNN k = 20"],
    ["KNN50", make_pipeline(preprocessing.StandardScaler(), PCA(n_components=PCA_COMPONENT_NO), KNeighborsClassifier(n_neighbors=50)), "KNN k = 50"] 
    
]


def getClassifierByName(name):
    for classifier in classifier_dictionary:
        if classifier[0] == name:
            return classifier

			
def readAllDataset1():
	###### Read all datasets ############
	trainData, trainLabel, testData = Utils.getDataset1()
	trainData, trainLabel, testData = Utils.getDataset2()
	trainData, trainLabel, testData = Utils.getDataset3()
	trainData, trainLabel, testData = Utils.getDataset4()
	trainData, trainLabel, testData = Utils.getDataset5()

	
def splitDataSet(trainData, trainLabel, testDataPercentage):
	trainingFeatures, testFeatures, trainingLabel, testLabel = train_test_split(trainData, trainLabel, test_size=testDataPercentage, random_state=0)
	return trainingFeatures, testFeatures, trainingLabel, testLabel

def trainModelWithDataset(trainData, trainLabel, testData, predictedLabelFile):
	givenTestFeatures = testData
	testDataPercentage = 0.4 # 40% 	
	accuracyVector = []
	
	for i in classifier_dictionary:	
		bestAccuracyAmongKFold = 0
		try:			
			for k in NO_OF_SPLITS:
				kFold = StratifiedKFold(n_splits = k, shuffle = True, random_state = SEED)
				
				classifier = i[1]
				scores = cross_val_score(classifier, trainData, trainLabel, cv = kFold, scoring='accuracy')
				print(scores)      				
					
				print("Accuracy with scaled data: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
				
				bestAccuracyAmongKFold = max(scores.mean(), bestAccuracyAmongKFold)								
						
			
			#print(bestAccuracyAmongKFold)			
		except IOError:
			print('An error occurred trying to read the file.')

		except ValueError:
			print('Non-numeric data found in the file.')

		except ImportError:
			print "NO module found"

		except EOFError:
			print('EOFError')

		except KeyboardInterrupt:
			print('You cancelled the operation.')

		accuracyVector.append(bestAccuracyAmongKFold)
		
			
	#np.where(a==a.max())
	print("accuracyVector :: ")
	print(accuracyVector)
	
	print("highest accurate classifier :: ")
	
	print(classifier_dictionary[np.argmax(accuracyVector, axis=0)][0])
	print("highest accuracy :: ")
	print(np.amax(accuracyVector, axis=0))
	mostAccurateClassifier = classifier_dictionary[np.argmax(accuracyVector, axis=0)][1]
	
	mostAccurateClassifier.fit(trainData, trainLabel)
	preditedLabel = mostAccurateClassifier.predict(givenTestFeatures)
	data = preditedLabel.reshape(preditedLabel.shape[0], 1)
	Utils.write_to_file(predictedLabelFile, data)
		
	return mostAccurateClassifier
	
	
################ Initialize Classification ###############

trainData, trainLabel, testData = Utils.getDataset1()	
predictedLabelFile = "../results/Classification1.txt"
trainModelWithDataset(trainData, trainLabel, testData, predictedLabelFile)


trainData, trainLabel, testData = Utils.getDataset2()	
predictedLabelFile = "../results/Classification2.txt"
trainModelWithDataset(trainData, trainLabel, testData, predictedLabelFile)


trainData, trainLabel, testData = Utils.getDataset3()	
predictedLabelFile = "../results/Classification3.txt"
trainModelWithDataset(trainData, trainLabel, testData, predictedLabelFile)

trainData, trainLabel, testData = Utils.getDataset4()	
predictedLabelFile = "../results/Classification4.txt"
trainModelWithDataset(trainData, trainLabel, testData, predictedLabelFile)



trainData, trainLabel, testData = Utils.getDataset5()	
predictedLabelFile = "../results/Classification5.txt"
trainModelWithDataset(trainData, trainLabel, testData, predictedLabelFile)


