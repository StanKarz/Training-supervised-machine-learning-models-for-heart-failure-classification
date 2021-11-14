# Training and evaluating machine learning models for heart disease classification

This Jupyter notebook aims to show how to train 4 different classification models namely logistic regression, decision tree classifier, random forest classifier, XGBoost on a heart disease dataset, with the goal to find out which model is the best at predicting the probability of heart disease occurence. Each model is trained on 60% of the original dataset and hyperparameters are tuned where necessary in an attempt to get the best version of a particular supervised learning algorithm. Upon training and evaluating all 4 models, K-Fold cross validation is utilised to discover which model performs best in terms of our evaluation metrics: ROC AUC score and accuracy on 5 different folds of the data. 


## Getting Started         
You can the find documentation for installing Jupyter [here](https://jupyter.org/install) or for more detailed instructions on installation [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) 

For local installation, make sure you have pip installed and run:

`
pip install notebook
`

## Usage - Running Jupyter notebook

### Running locally

launch with

`
$ jupyter notebook
`

### Running remotely

You will need to make some configurations before running the Jupyter notebook remotely, see [docs](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html)

## References


* [Numpy](https://numpy.org/) - NumPy is an open source project aiming to enable numerical computing with Python
* [Pandas](https://pandas.pydata.org/) - Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool built on top of Python
* [Matplotlib](https://matplotlib.org/) - Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python
* [Seaborn](https://seaborn.pydata.org/) - Seaborn is a Python data visualization library based on matplotlib
* [sk-learn](https://scikit-learn.org/stable/) - sk-learn is a machine learning library for the Python programming language
* [Heart disease dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction) - Kaggle is is an online community of data scientists and machine learning practitioners
