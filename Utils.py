import numpy as np
import pandas as pd

from numpy import genfromtxt

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

NO_OF_SPLITS = [3, 5, 10]
SEED = 9999999

def getDataset1():
    
	trainDataFile = "../dataset/TrainData1.txt"
	trainLabelFile = "../dataset/TrainLabel1.txt"
	testDataFile = "../dataset/TestData1.txt"

	trainData = genfromtxt(trainDataFile, delimiter=",")
	trainLabel = genfromtxt(trainLabelFile, delimiter="\n")
	testData = genfromtxt(testDataFile, delimiter=",")
	
	return trainData, trainLabel, testData

def getDataset2():
    
	trainDataFile = "../dataset/Imputed_TrainData2.txt"
	trainLabelFile = "../dataset/TrainLabel2.txt"
	testDataFile = "../dataset/TestData2.txt"

	trainData = genfromtxt(trainDataFile, delimiter="\t")
	trainLabel = genfromtxt(trainLabelFile, delimiter="\n")
	testData = genfromtxt(testDataFile, delimiter="\t")
	
	return trainData, trainLabel, testData

def getDataset3():
    
	trainDataFile = "../dataset/TrainData3.txt"
	trainLabelFile = "../dataset/TrainLabel3.txt"
	testDataFile = "../dataset/TestData3.txt"

	
	with open(trainDataFile, 'r') as f:
		num_cols = len(f.readline().split())
		f.seek(0)
		data = genfromtxt(f, usecols = range(0,num_cols))
	
	trainData = data
	trainLabel = genfromtxt(trainLabelFile, delimiter="\n")
	
	with open(testDataFile, 'r') as f:
		num_cols = len(f.readline().split())
		f.seek(0)
		data = genfromtxt(f, usecols = range(0,num_cols))
		
	testData = data

	return trainData, trainLabel, testData

	
def getDataset4():
        
	trainDataFile = "../dataset/Imputed_TrainData4.txt"
	trainLabelFile = "../dataset/TrainLabel4.txt"
	testDataFile = "../dataset/TestData4.txt"
	
	trainData = genfromtxt(trainDataFile, delimiter="\t")
	trainLabel = genfromtxt(trainLabelFile, delimiter="\n")
	testData = genfromtxt(testDataFile, delimiter=",")
	
	return trainData, trainLabel, testData


def getDataset5():
    
	trainDataFile = "../dataset/TrainData5.txt"
	trainLabelFile = "../dataset/TrainLabel5.txt"
	testDataFile = "../dataset/TestData5.txt"
	
	with open(trainDataFile, 'r') as f:
		num_cols = len(f.readline().split())
		f.seek(0)
		data = genfromtxt(f, usecols = range(0,num_cols))
	
	trainData = data
	
	trainLabel = genfromtxt(trainLabelFile, delimiter="\n")
	testData = genfromtxt(testDataFile, delimiter=" ")
	
	return trainData, trainLabel, testData


def write_to_file(filename, array):

    with open(filename, "w") as f:
        for l in array:
            f.write(str(int(l)) + "\n")
				
	
	