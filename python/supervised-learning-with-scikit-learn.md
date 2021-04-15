# Supervised Learning with scikit-learn

- [Supervised Learning with scikit-learn](#supervised-learning-with-scikit-learn)
  - [Introduction to Supervised Learning](#introduction-to-supervised-learning)
    - [Types of Machine Learning](#types-of-machine-learning)
    - [Supervised Learning](#supervised-learning)
  - [Classification](#classification)
    - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Classification](#classification-1)
  - [Regression](#regression)
    - [Cross Validation](#cross-validation)
    - [Regularised Regression](#regularised-regression)
  - [Fine tuning a model](#fine-tuning-a-model)
    - [Measuring model performance](#measuring-model-performance)
    - [Logistic Regression and the ROC curve](#logistic-regression-and-the-roc-curve)
    - [Area under the ROC curve (AUC)](#area-under-the-roc-curve-auc)
    - [Hyperparameter tuning](#hyperparameter-tuning)
  - [Preprocessing data](#preprocessing-data)
    - [Dealing with categorical features](#dealing-with-categorical-features)
    - [Handling missing data](#handling-missing-data)
    - [Centering and Scaling](#centering-and-scaling)
  - [Bringing it all together](#bringing-it-all-together)
    - [Pipeline for Classification](#pipeline-for-classification)
    - [Pipeline for Regression](#pipeline-for-regression)

## Introduction to Supervised Learning

### Types of Machine Learning

> **Machine Learning**: the art and science of giving computers the ability to learn to make decisions from data, without being explicityly programmed.

Examples:
- Learning to predict whether an email is spam or not
- Clustering wikipedia entries into different categories

**Supervised Learning**: Uses labelled data
- Predictor variables/features and a target variable
- Data organised with row per data point and column per feature
- Aim: Predict the target variable, given the predictor variables
  - Classification- Target variable categoric
  - Regression- Target variable continuous

**Unsupervised learning**: Uses unlabelled data
- Uncovering hidden patterns from unlabelled data
- Example: grouping customers into distinct categories (Clustering)

**Reinforcement Learning**:
- Software agents interact with an environment
- Learn how to optimise behaviour, given a system of rewards and punishments
- Draws inspiration from behavioural psychology
- Applications: Economics, Genetics, game playing.
- In 2016 Google trained AlphaGo to be the first computer to defeat the world champion in Go.

### Supervised Learning

**Naming conventions:**

Features = Predictor Variables = Independent Variables

Target Variable = Dependent Variable = Response Variable

**Uses:**
- Automate time-consuming or expensive manual tasks
  - Example: Doctor's diagnosis
- Make predictions about the future
  - Example: Will a customer click on an ad or not?
- Need labelled data
  - Historical data with labels
  - Experiments to get labelled data (e.g. AB testing)
  - Crowd-sourcing labeled data

For this course we wil use the `scikit-learn` Python library.
- Integrated well with the SciPy stack (numpy etc.)

Other libraries:
- TensorFlow
- keras

## Classification

### Exploratory Data Analysis

```python
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib as plt
plt.style.use('ggplot')
iris = datasets.load_iris()

type(iris)
> sklearn.datasets.base.Bunch # Key-value pairs, similar to dict

iris.keys()
> dict_keys(['data', 'target_names', 'DESCR', 'feature_names', 'target'])

type(iris.data), type(iris.target)
> (numpy.ndarray, numpy.ndarray)
# Both feature and target data provided as numpy arrays
```

Performing EDA on datasets:

```python
X = iris.data,
Y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df.head()
```

```python
_ = pd.plotting.scatter_matrix(df, c=y, figsize=[8,8], s=150, marker='D')
```

### Classification

**Scikit-learn fit and predict**:
- All machine learning models in scikit-learn are implemented as Python classes.
  - implement the algorthms for learning
  - predict and store the information learned from the data
- Training a model on the data = 'fitting' a model to the data
  - `.fit()` method
- To predict the labels of new data: `.predict()` method

The scikit-learn API requires data to be in a pandas DataFrame or numpy array, features take on continuous variables and no missing values in the data.

**k-Nearest Neighbors**:
- Predict data point by looking at 'k' closest labelled data points and taking the majority vote

```python
from sklearn.neighbours import KNeighbourClassifier

knn = KNeighbourClassifier(n_neighbours=6)
knn.fit(iris['data'], iris['target'])
```

Predicting unlabelled data:
```python
X_new = np.array(
            [[5.6, 2.8, 3.9, 1.1],
             [5.7, 2.6, 3.8, 1.3],
             [4.7, 3.2, 1.3, 0.2]])

prediction = knn.predict(X_new)

print('Prediction: {}'.format(prediction))

> Prediction: [1 1 0] # versicolor for first two, setosa for third
```

**Measuring Model Performance:**

- Accuracy = Fraction of correct predictions
- Testing the accuracy of a model on the data that was used to train it will not be indicative of its ability to generalise- it is, therefore, common to split data into *training* and *test* sets. 

```python
from sklearn.model_selection import train_test_split

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Fit with KNN and make prediction on test set
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Test set predictions:\\n {}".format(y_pred))

# Evaluate accuracy of test prediction vs known labels
knn.score(X_test, y_test)
```

- `test_size`: proportion of data to be allocated to test set.
- `random_state`: seed for random generator to split data
- `X_train`: training data
- `X_test`: test data
- `y_train`: training labels
- `y_test`: test_labels
- `stratify=y`: distributes labels in train and test sets as they are in the original dataset

**Model complexity:**
- Larger K $\implies$ Smoother decision boundary $\implies$ Less complex model
- Smaller K $\implies$ More comples model $\implies$ Can lead to overfitting

There is an optimum number of neighbors which maximises accuracy on the test set, too small K will overfit the data and two small will underfit it.

## Regression

```python
boston = pd.read_csv('boston.csv')

X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values

X_rooms = X[:,5]

y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show()
```

```python
import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_rooms, y)

prediction_space = np.linspace(min(X_rooms,
                               max(X_rooms)
                               ).reshape(-1, 1)

plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space,
         reg.predict(prediction_space),
         color='black',
         linewidth=3)
         plt.show()
```

**Mechanics of linear regression:**

In two dimensions we want to fit a line $y=ax+b$, where $y$ is the target, $x$ is the single feature and $a$, $b$ are the parameters of our model.

A common method is to define a 'loss function' and then choose the line that minimises this function.

**OLS (Ordinary Least Squares)**: Minimising the sum of the squares of the residuals (distance between data point and line)

In higher dimensions linear regression generalises to a variable $a_i$ for each feature $x$ as well as the variable $b$:
> $y=a_1x_x+a_2x_2+...+a_nx_n+b$

The sckit-learn API works exactly the same for higher dimensions, we simply pass two arrays: features and target.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
                                       X, y,
                                       test_size=0.3,
                                       random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
reg_all.score(X_test, y_test)
```

Note: Generally will never use linear regression out of the box in this way, we will more likely want to use regularisation which places further constraints on model coefficients.

### Cross Validation

Since the results predicted by our model are dependent on how we split our data, i.e the test set may not may not be representative of the models ability to generalise, we use **cross-validation** to combat this.

To do this we split the data into $k$ 'folds'. We take the first fold as the test set and fit our model on the remaining four, computing the metric of interest. We then repeat this for the second fold as the test set and so on up to the $k$th fold. We can then compute statistics such as the mean, median and 95% confidence intervals.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=5)

print(cv_results)
print(np.mean(cv_results))
```

### Regularised Regression

If we allow arbitrarily large coefficients we can risk overfitting our regression models. For this reason, it is common practice to alter the loss function such that it penalises for large coefficients.This is called regularisation.

**Ridge regression:**

$\text{Loss function} = \text{OLS loss function} + \alpha*\sum_{i=1}^{n}{a_i^2}$

Applying a loss function which includes a term multiplying some parameter $\alpha$ by the sum of the squares of the coefficient penalises large positive/negative coefficients from being selected by the model.

Choosing $\alpha$ for ridge regression is similar to choosing $k$ in KNN and is known as **hyperparameter tuning**. $\alpha$ (also sometimes known as lambda) controls model complexity. An alpha of zero reduces back to OLS and doesn't account for overfitting, whereas a large $\alpha$ means that large coefficients are significantly penalised and can lead to selecting a model which is too simple, underfitting the data.

```python
from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(
                                       X, y,
                                       test_size=0.3
                                       random_state=42)

ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)
```

**Lasso regression:**

$\text{Loss function} = \text{OLS Loss Function} + \alpha*\sum_{i=1}^{n}{|a_i|}$

```python
from sklearn.linear_model import Lasso

X_train, X_test, y_train, y_test = train_test_split(
                                       X, y,
                                       test_size=0.3
                                       random_state=42)

lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)
```

Lasso regression is useful for selecting important features of a dataset, since it tends to shrink coefficients of less important features to zero, the features not shrunk to zero are then selected.

```python
from sklearn.linear_model import Lasso

names = boston.drop('MEDV', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_

_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.y_label('Coefficients')
plt.show()
```

Plotting the Lasso coefficients as a function of feature name shows clearly the relative importance of various features.

> Lasso regression is great for feature selection, but when building regression models, Ridge regression should be first choice.

```python
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space,
                    cv_scores + std_error,
                    cv_scores - std_error,
                    alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
```

## Fine tuning a model

### Measuring model performance

**Accuracy**: Fraction of correctly classified objects

Accurancy is not always a useful metric. Consider a spam classification problem in which 99% of emails are real and 1% are spam. A classifier which predicts all emails are real would be 99% accurate.

This issue of one class being more frequent than another is known as class imbalance and requires a more nuanced metric to evaluate the performance of our model.

**Confusion Matrix**:

|                    | Predicted Spam Email | Predicted Real Email |
| ------------------ | -------------------- | -------------------- |
| **Actual: Spam Email** | True Positive        | False Negative       |
| **Actual: Real Email** | False Positive       | True Negative        | 

From the confusion matrix, we can easily compute several other metrics:

$\text{Accuracy} = \frac{t_p+t_n}{t_p+t_n+f_p+f_n}$

$\text{Precision\textbackslash Positive Predictive Value (PPV)} = \frac{t_p}{t_p+f_p}$

- High precision $\implies$ Not many real emails predicted as spam

$\text{Recall\textbackslash Sensitivity} = \frac{t_p}{t_p+f_n}$

- High Recall $\implies$ Predicted most spam emails correctly

$\text{F1score} = 2 \cdot \frac{precision*recall}{precision+recall}$

```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

knn = KNeighborsClassifiers(n_neighbors=8)
X_train, X_test, y_train, y_test = train_test_split(
                                       X, y,
                                       test_size=0.4,
                                       random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))

> [[52 7]
    3 112]]
``
print(classification_report(y_test, y_pred)

>            precision  recall  f1_score  support
          0       0.95    0.88      0.91       59 
          1       0.94    0.97      0.96      115
  avg/total       0.94    0.94      0.94      174
```

For all metrics in scikit learn, the first argument is always the true label and then prediction is second.

### Logistic Regression and the ROC curve

Logistic regression outputs probabilities. If the probability is greater then 0.5, the data is labelled as 1, else the data is labelled as 0.

$p>0.5 \implies 1$

$p<0.5 \implies 0$

Logistic regression produces a linear decision boundary for classification problems.

```python
from sklearn.linear_model import Logistic Regression
from sklearn.model_selection import train_test_split

logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(
                                       X, y,
                                       test_size=0.4,
                                       random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
```

**Probability Thresholds:**

By default logistic regression uses a threshold of `0.5`.
k-NN classifiers also have thresholds which may be varied.

If we vary the threshold and consider the true positive and false positive rates, we get a curve known as the **receiver operating characteristic curve** or **ROC** curve. A threshold of zero predicts `1` for all the data, while a threshold of 1 predicts `0` for all data, the ROC curve describes the behaviour between these two extremes.

![ROC Curve](https://upload.wikimedia.org/wikipedia/commons/3/36/Roc-draft-xkcd-style.svg)

```python
from sklearn.metrics import roc_curve

y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```

### Area under the ROC curve (AUC)

Larger area under the model =  better model

The area under the ROC is a useful metric of the performance of a model, since a larger area implies a more successful model.

```python
from sklearn.metrics import roc_auc_score

logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(
                                      X, y,
                                      test_size=0.4,
                                      random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)
```

**AUC using cross-validation:**
```python
from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(
              logreg, X, y, cv=5,
              scoring='roc_auc')
```

### Hyperparameter tuning

Linear regression: Choosing parameters
Ridge/lasso regression: Choosing alpha
k-Nearest Neighbors: Choosing n_neighbors

**Hyperparameters**: Parameters that cannot be explicitly learned by the model, i.e. must be chosen by the creator of the model.

**Hyperparameter tuning**: Testing a range of hyperparameter values and evaluating their performance in order to choose the best one.

It is important when hyperparameter tuning that we use **cross-validation** so as not to overfit the hyperparameter to the test set.

**Grid search cross validation**:

Grid search cross-validation uses a grid of all possible combinations of the hyperparameters we are selecting from. We then perform k-fold cross-validation for each point in the grid and choose the combination of parameters that performs best.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)

knn_cv.best_params_
> {'n_neighbors': 12}

knn_cv.best_score_
> 0.933216
```

Since `GridSearchCV` can be computationally expensive, especially when searching over a large hyperparameter space and with multiple hyperparameters. We can instead use `RandomizedSearchCV` which takes a fixed number hyperparameter settings are sampled from specified probability distributions.

```python
# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
```

**Hold out set:**

Since it is important that we know how well our model performs on an unseen dataset, it is important to split our data at the very begining into a training set and a **hold-out set**. We can then perform cross-validation on the training set to tune the model's hyperparameters and then use the hold-out set with to test the model against an unseen dataset.

## Preprocessing data

### Dealing with categorical features

Scikit-learn will not accept categorical features by default. We can deal with this by splitting a categorical feature into a number of binary features called 'dummy variables', one for each category.

**Categoric Dataset**:

| Origin |
| ------ |
| US     |
| Europe |
| Asia   |

**Binary features:**

| origin_Asia | origin_Europe | origin_US |
| ----------- | ------------- | --------- |
| 0           | 0             | 1         |
| 0           | 1             | 0         |
| 1           | 0             | 0         |


**Encoded dataset:**

Note that in the above table if a car is not from the US and not from Asia then implicitly it must be from Europe/ Therefore, including an additional column for Europe is duplicating information which may present an issue for some models. To avoid this we can exclude this additional column.

| origin_Asia | origin_US |
| ----------- | --------- |
| 0           | 1         |
| 0           | 0         |
| 1           | 0         |

**Dealing with categorical features in Python:**

- scikit-learn: `OneHotEncoder()`
- pandas: `get_dummies()`

| mpg | displ | hp | weight | accel | origin | size |
| --- | ----- | -- | ------ | ----- | ------ | ---- |
| 18.0| 250.0 | 88 | 3139   | 14.5  | US     | 15.0 |
| 9.0 | 304.0 | 193| 4732   | 18.5  | Europe | 20.0 |
| 36.1| 91.0  | 60 | 1800   | 16.4  | Asia   | 10.0 |

```python
import pandas as pd

df = pd.read_csv('auto.csv')
df_origin = pd.get_dummies(df)
df_origin.drop('origin_Asia', axis=1)
```

| mpg | displ | hp | weight | accel | size | origin_Europe | origin_US |
| --- | ----- | -- | ------ | ----- | ---- | ------------- | --------- |
| 18.0| 250.0 | 88 | 3139   | 14.5  | 15.0 | 0             | 1         |
| 9.0 | 304.0 | 193| 4732   | 18.5  | 20.0 | 1             | 0         |
| 36.1| 91.0  | 60 | 1800   | 16.4  | 10.0 | 0             | 0         |

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ridge = Ridge(alpha=0.5, normalize=True).fit(X_train, y_train)

ridge.score(X_test, y_test)
```

### Handling missing data

We say that data is missing where there is no value for a given feature in a particular row.

One method of dealing with missing values is to drop this data from the dataset. However, this may lead to the loss of significant amounts of data.

```python
df.feature_a.replace(0, np.nan, inplace=True)
df.feature_b.replace(0, np.nan, inplace=True)

df = df.dropna()
```

**Imputing missing data:**

An alternative method of dealing with missing values is to make an educated guess of their values. For examples, taking the mean of non-missing entries.

```python
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN' strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)
```

**Imputing within a pipeline:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
logreg = LogisticRegression()

steps = [('imputation', imp),
         ('logistic_regression', logreg)]

pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
pipeline.score(X_test, y_test)
```

Note that each step in a pipeline must be a transformer and the last must be an estimator, such as a classifier or regressor.

### Centering and Scaling

Many machine learning models use some form of distance measure. Therefore, unless normalised, also known as scaling ans centring, features on larger scales may unduly influence models.

**Methods of normalisation:**

- Subtract the mean and divide by variance (**Standardisation**). This centres all features around zero, with variance one.
- Subtract the mimimum and divide by the range $\implies$ minimum zero, maximum one.
- Can also normalise ranges from -1 to +1

**Standardisation in scikit-learn:**

```python
from sklearn.preprocessing import scale

X_scaled = scale(X)
```

**Scaling in a pipeline:**

```python
from sklearn.preprocessing import StandardScaler

steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

knn_scaled = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy_score(y_test, y_pred)
```

**CV and scaling in a pipeline:**

```python
steps = [('scaler'), StandardScaler()),
         ('knn'), KNeighborsClassifier())]
pipeline = Pipeline(steps)

parameters = {knn__n_neighbors: np.arange(1, 50)}

X_train, X_test, y_train, y_test = train_test_split(
                                      X, y,
                                      test_size=0.2,
                                      random_state=21)

cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
```

## Bringing it all together

### Pipeline for Classification

```python
# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(
                                      X, y,
                                      test_size=0.2,
                                      random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
```

### Pipeline for Regression

```python
# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y,
                                    test_size=0.4,
                                    random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
```