# Tunisian Baccalaureat Orientation System

In this project, I tried to implement an intelligent system which has as mission to guide tunisian students in the post baccalaureate orientation. 

A system that will be based on the personalities of the students to determine the most suitable orientation for them. It will therefore reduce the uncertainty of students who will be able to make their decisions while taking into account their characters. This platform will ask psychological questions and thanks to an intelligent model, it will be able to recommend the best field of study. 

## Data Gathering

The first step was to garther the data needed for the machine learning model to learn, a form which proposes psychological questions was filled by 125 students in various fields of study. I targeted 10 fields : Humanities - Media - Legal science - Arts and Crafts - Medical - Management and Arts & Crafts - Medical - Management & Economics - Fundamental Science - Agriculture & Environment - Engineering Studies OR Technological Studies.

## Data Preprocessing

### Null values imputing

Questions with a lot of null values were deleted, questions with a less of 5% of null values were imputed using the most frequent value imputation technique.

### Data Encoding

Data encoding is a crucial step when preparing data for the model. 
Two types of encoding were performed: 

* Label Encoding 

The Label Encoding consists in assigning to the categorical values of a column integers that translate the order of magnitude between them. This encoding method is used to encode ordinal variables.

* One Hot Encoding

The one Hot encoding consists in transforming a column into n columns according to the number n of categorical values present in this column. Each created column corresponds to different values of the original column. The values of the created columns are binary. This encoding method is used to encode ordinal variables which have no order of magnitude between them

### Data Balancing

When working with unbalanced training set, the classifier has only few examples of the minority class to learn from. It will therefore be biased towards the majority population and produces predictions that are potentially less robust than in the absence of imbalance. 

To compensate for this imbalance, we used the random oversampling technique which consists in supplementing the training data with multiple copies of random instances of the minority class.

### Model Selection 

I chose to implement 3 algorithms and compare their performances : logistic regression, Support Vector Machine (SVM) and the multilayer perceptron algorithm (MLP) which represents a type of artificial neural network.Then, I developed an algorithm that relies on a majority voting mechanism between the 3 other algorithms already mentioned.

### Model Validation 

k-fold cross validation 

The original sample is divided into k blocks, and then one of the k blocks is selected as the validation set while the other k-1 blocks represent the training set. 
After learning, a validation performance can be calculated. 
Then the operation is repeated by selecting another validation sample among the predefined blocks. At the end of the procedure we obtain k performance scores, one per block. The mean and standard deviation of the k performance scores can be computed to estimate the overall average bias and variance.


### Web App Development

To create the web interface of our project, I used the Streamlit open source Python framework which is used to make for beautiful and powerful machine learning data applications.

This interface consists of collecting the user's answers to the previously mentioned questions and according to which an academic post-bac field will be recommended along with its description with best universities in Tunisia in which the student can study it.
