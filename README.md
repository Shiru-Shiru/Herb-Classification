# Herb-Classification

## Overview
Herb Classification using Machine Learning

## Dataset
You can download the dataset for this project here: [Herb_Image](https://drive.google.com/drive/folders/1JkqctNZxmyTUODpSuwdCIU0cKw_6zLyK?usp=sharing)

## Libraries Used
The following Python libraries are utilized in this project:

- `numpy`: For numerical computations and array handling.
- `matplotlib.pyplot`: To create visualizations of the data and model performance.
- `sklearn.decomposition` (PCA): For principal component analysis to reduce data dimensionality.
- `sklearn.manifold` (TSNE): For t-SNE visualizations to observe clustering.
- `sklearn.svm` (SVC, LinearSVC): Support Vector Classifier models for classification.
- `sklearn.tree` (DecisionTreeClassifier): Decision tree classifier for classification tasks.
- `sklearn.ensemble` (RandomForestClassifier, AdaBoostClassifier): Ensemble methods for boosting and random forest classifications.
- `sklearn.neighbors` (KNeighborsClassifier): K-Nearest Neighbors classifier for classifying based on proximity.
- `sklearn.datasets` (load_digits): To load example datasets for testing.
- `sklearn.model_selection` (train_test_split, cross_val_score, GridSearchCV): For splitting data, cross-validation, and hyperparameter tuning.
- `sklearn.metrics` (accuracy_score, confusion_matrix, ConfusionMatrixDisplay): For model evaluation and accuracy metrics.
- `skimage.transform` (resize): For resizing image data.
- `skimage.io` (imread): For reading and handling image files.
- `cv2`: OpenCV library, used for additional image processing tasks.
- `pandas`: For data manipulation and data frame handling.
- `sklearn.preprocessing` (StandardScaler): To standardize features by removing the mean and scaling to unit variance.
- `seaborn`: For enhanced data visualization, particularly for visualizing model metrics.

## Data Source
The herb images for this project are sourced from the [Oriental Medicinal Herb Images](https://www.kaggle.com/datasets/trientran/oriental-medicinal-herb-images/data) dataset on Kaggle.

## Acknowledgments
Special thanks to [Triet Tran](https://www.kaggle.com/trientran) for providing the **Oriental Medicinal Herb Images** dataset on Kaggle, which serves as the basis for this classification project.




