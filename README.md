# Classification
 In this project I've applied couple of supervised learning algorithms for classifying newly arrived microarray gene expression data based on given training dataset and training labels. I've used LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, SVC, and KNeighborsClassifier.

My two key approaches to Classification:

1. Determine Accuracy of classification:
To measure accuracy of classification I have to use multiple classifier and cross validate their classification results based on multiple metrics like different sized train and test data, using scaled/un-scaled dataset, perform feature selection, etc.

2. Perform Prediction using most accurate Classifier:
I take the classifier that has highest accuracy and perform classification on given test dataset.
Write the prediction results into files.
Repeat steps of Determine Accuracy of classification for each given train datasets and labels

#### Environment Setup
a. Install numpy, scipy, sci-kit learn, pandas (I used numpy==1.13.3, pandas==0.21.0, scikit-learn==0.19.1, scipy==1.0.0). See http://scikit-learn.org/stable/
		
##### Set up your working directory
The working directory should contain:
a. All training data, training label, and test data files are in a sub directory named 'Dataset'
- TrainData 
- TrainLabel
- TestData 
				
b. Source codes [in a sub directory named 'SourceCode']
- classifiers.py 
	[Estimate the accuracy of multiple classifiers with different data preposessing techniques]
- Utils.py
	[Consists some utility functions for file reading/writting]

##### Run the source code files
a. Nevigate to your working directory where the source files are
b. Run command in Windows/Linux command prompt:
```sh
>>> python Utils.py
>>> python classifiers.py
```			
		
##### Expected Outputs 
All predicted labels will be found under "Results" directory
			