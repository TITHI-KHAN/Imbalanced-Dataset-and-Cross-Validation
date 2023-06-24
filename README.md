# Imbalanced-Dataset-&-Cross-Validation

![image](https://github.com/TITHI-KHAN/Imbalanced-Dataset-and-Cross-Validation/assets/65033964/6ddb7ae3-b314-4135-affa-71238d235875)

#   Machine Learning Model

![image](https://github.com/TITHI-KHAN/Imbalanced-Dataset-and-Cross-Validation/assets/65033964/5d44c3ce-b308-4720-939c-4638c691f084)

# Cross Validation
Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. It is a technique used to protect against overfitting in a predictive model, particularly in a case where the amount of data may be limited. In cross-validation, you make a fixed number of folds (or partitions) of the data, run the analysis on each fold, and then average the overall error estimate.

![image](https://github.com/TITHI-KHAN/Imbalanced-Dataset-and-Cross-Validation/assets/65033964/96427114-ed12-4788-ab73-487d80edcc4c)

# Cross Validation Techniques

▪ Hold Out Cross Validation

▪ K-Fold Cross Validation

▪ Leave One-Out Cross Validation (LOOCV)

▪ Stratified K Fold Cross Validation

# Hold Out Cross Validation

![image](https://github.com/TITHI-KHAN/Imbalanced-Dataset-and-Cross-Validation/assets/65033964/ff934f0d-b4c7-4001-9883-a1e8b292fdcc)

Hold-out cross-validation is a simple and commonly used technique for assessing the performance of a machine learning model. It involves splitting the available data into two sets: a training set and a validation (or testing) set. The model is trained on the training set and then evaluated on the validation set to estimate its performance.

Here's how hold-out cross-validation works:

1. **Data Splitting**: The available data is divided into two sets: a training set and a validation set. The typical split is around 70-80% for training and 20-30% for validation, but this can vary depending on the size of the dataset and the specific requirements of the problem.

2. **Model Training**: The machine learning model is trained on the training set using the available features and corresponding labels (or target values). The model learns the underlying patterns and relationships in the training data.

3. **Model Evaluation**: Once the model is trained, it is evaluated on the validation set. The model predicts the labels (or target values) for the samples in the validation set, and the predicted values are compared with the actual values. Various evaluation metrics, such as accuracy, precision, recall, or mean squared error, can be used to assess the model's performance.

4. **Performance Estimation**: The evaluation results on the validation set provide an estimate of the model's performance on unseen data. This estimation helps in understanding how well the model is likely to perform on new, unseen samples.

5. **Iterative Process**: Hold-out cross-validation can be performed iteratively by repeating the process with different random splits of the data. This helps in obtaining a more robust estimate of the model's performance by averaging the results across multiple splits.

It is important to note that hold-out cross-validation has some limitations. It may lead to high variance in the estimated performance, especially when the dataset is small. In such cases, more advanced techniques like k-fold cross-validation or stratified sampling can provide more reliable performance estimates.

# K-Folds Cross Validation

K-fold cross-validation is a popular technique for evaluating the performance of machine learning models. It provides a more reliable estimate of a model's performance by dividing the available data into k subsets, or folds. The model is trained and evaluated k times, with each fold serving as the validation set once while the remaining folds are used for training.

Here's how K-fold cross-validation works:

1. **Data Splitting**: The available data is divided into k subsets or folds of approximately equal size. Typically, k is chosen as a value between 5 and 10, but it can vary depending on the size of the dataset and the specific requirements of the problem.

2. **Model Training and Evaluation**: The model is trained and evaluated k times. In each iteration, one of the k folds is used as the validation set, and the remaining k-1 folds are combined to form the training set. The model is trained on the training set and then evaluated on the validation set.

3. **Performance Aggregation**: The performance of the model is recorded for each iteration. Common evaluation metrics, such as accuracy, precision, recall, or mean squared error, can be calculated and aggregated across all k iterations to obtain an overall performance estimate.

4. **Performance Comparison**: The performance estimates from the k iterations can be used to compare different models or different configurations of the same model. This helps in selecting the best model or hyperparameters that generalize well to unseen data.

5. **Bias-Variance Tradeoff**: K-fold cross-validation helps in assessing the bias-variance tradeoff of a model. If the model performs consistently well across all k folds, it indicates low bias. On the other hand, if the model shows high variance in performance across the folds, it suggests overfitting or sensitivity to the specific training-validation splits.

K-fold cross-validation provides a more robust estimate of a model's performance compared to hold-out validation since it utilizes the entire dataset for both training and evaluation. It helps in reducing the impact of the specific training-validation split on the performance estimate and provides a more representative evaluation of the model's generalization ability.

The general procedure is as follows:
1. Shuffle the dataset randomly.
   
2. Split the dataset into k groups.
   
3. For each unique group:
   
        1. Take the group as a hold out or test data set.
           
        2. Take the remaining groups as a training data set.
           
        3. Fit a model on the training set and evaluate it on the test set.
           
        4. Retain the evaluation score and discard the model.
     
4. Summarize the skill of the model using the sample of model evaluation scores.

![image](https://github.com/TITHI-KHAN/Imbalanced-Dataset-and-Cross-Validation/assets/65033964/ed63a087-49e2-43eb-86d7-1e8354997f2f)

![image](https://github.com/TITHI-KHAN/Imbalanced-Dataset-and-Cross-Validation/assets/65033964/141d8edf-3d09-4ff1-a6dd-06128bf84f9f)


# Leave One-Out Cross Validation (LOOCV)

![image](https://github.com/TITHI-KHAN/Imbalanced-Dataset-and-Cross-Validation/assets/65033964/387cfe6d-84dd-4b72-86ca-5b280d04bcda)

Leave One-Out Cross Validation (LOOCV) is a specific variation of cross-validation where the number of folds is equal to the number of samples in the dataset. In LOOCV, each sample is used as a validation set once, and the remaining samples are used for training the model. LOOCV is an exhaustive cross-validation technique and provides an unbiased estimate of the model's performance.

Here's how LOOCV works:

1. **Data Splitting**: For LOOCV, each sample in the dataset is considered as a separate fold. This means that if you have n samples in your dataset, you will create n folds, where each fold contains a single sample for validation, and the remaining n-1 samples are used for training.

2. **Model Training and Evaluation**: The model is trained and evaluated n times, with each iteration using a different sample as the validation set. In each iteration, the model is trained on the n-1 samples and then evaluated on the single sample that was left out.

3. **Performance Aggregation**: The performance of the model is recorded for each iteration. Evaluation metrics, such as accuracy, precision, recall, or mean squared error, can be calculated and aggregated across all n iterations to obtain an overall performance estimate.

4. **Performance Comparison**: The performance estimates from the LOOCV iterations can be used to compare different models or different configurations of the same model. This helps in selecting the best model or hyperparameters that generalize well to unseen data.

LOOCV provides an unbiased estimate of a model's performance because each sample in the dataset is used for validation exactly once. This makes LOOCV particularly useful when working with small datasets or when the model's performance on each individual sample is crucial.

However, LOOCV can be computationally expensive, especially for large datasets, as it requires training and evaluating the model n times. Therefore, it might not be feasible to use LOOCV in such cases.

# Stratified K Fold Cross Validation

![image](https://github.com/TITHI-KHAN/Imbalanced-Dataset-and-Cross-Validation/assets/65033964/122c62c9-91c1-4a86-ab27-a02a31d294a7)

Stratified K-fold cross-validation is a variation of the K-fold cross-validation technique that takes into account the class distribution in the dataset. It ensures that each fold has a similar distribution of target classes, which is particularly useful when dealing with imbalanced datasets. Stratified K-fold cross-validation helps in obtaining more reliable performance estimates, especially when the class distribution is uneven.

Here's how Stratified K-fold cross-validation works:

1. **Data Splitting**: The available data is divided into k subsets or folds, similar to K-fold cross-validation. However, in Stratified K-fold, the splitting is performed in a way that maintains the same proportion of target classes in each fold as the original dataset.

2. **Model Training and Evaluation**: The model is trained and evaluated k times, similar to K-fold cross-validation. In each iteration, one of the k folds is used as the validation set, and the remaining k-1 folds are combined to form the training set. The model is trained on the training set and then evaluated on the validation set.

3. **Performance Aggregation**: The performance of the model is recorded for each iteration. Evaluation metrics, such as accuracy, precision, recall, or mean squared error, can be calculated and aggregated across all k iterations to obtain an overall performance estimate.

4. **Performance Comparison**: The performance estimates from the Stratified K-fold iterations can be used to compare different models or different configurations of the same model. This helps in selecting the best model or hyperparameters that generalize well to unseen data, considering the class distribution.

Stratified K-fold cross-validation is particularly useful when the dataset has imbalanced class distributions, where one or more classes have significantly fewer samples compared to others. It ensures that each fold contains a representative mix of samples from each class, which helps in obtaining performance estimates that are less biased by the class distribution.

# Synthetic Minority Oversampling Technique (SMOTETomek)

The Synthetic Minority Oversampling Technique (SMOTE) and Tomek Links (SMOTETomek) are two separate techniques commonly used in the field of imbalanced classification to address the issue of imbalanced datasets. While they are distinct techniques, they can be combined to create a hybrid sampling approach known as SMOTETomek.

1. **SMOTE (Synthetic Minority Oversampling Technique)**: SMOTE is an oversampling technique that aims to balance imbalanced datasets by generating synthetic samples for the minority class. It works by randomly selecting a sample from the minority class and generating synthetic samples along the line segments connecting it to its nearest neighbors. This helps in increasing the number of minority class samples and reducing the class imbalance.

2. **Tomek Links**: Tomek Links are pairs of samples from different classes that are closest to each other. These pairs of samples are considered to be potentially noisy or ambiguous, and removing them can help improve the decision boundary between classes. Tomek Links removal involves identifying Tomek Links pairs and removing the majority class sample from each pair.

3. **SMOTETomek**: SMOTETomek combines the oversampling technique of SMOTE with the undersampling technique of Tomek Links. It applies SMOTE to generate synthetic samples for the minority class, and then applies Tomek Links to identify and remove Tomek Links pairs between the minority and majority class samples. This approach aims to improve the class balance by oversampling the minority class and undersampling the majority class simultaneously.

By combining the strengths of both oversampling and undersampling techniques, SMOTETomek can potentially improve the performance of classifiers on imbalanced datasets. It can help mitigate the effects of class imbalance and lead to more robust and accurate predictions.

# Near Miss for Under_Sampling

Near Miss is an under-sampling technique commonly used in the field of imbalanced classification to address the issue of imbalanced datasets. It aims to reduce the imbalance between the majority and minority classes by selectively removing samples from the majority class.

The Near Miss algorithm identifies samples from the majority class that are considered "nearest" to the minority class samples. It selects a subset of majority class samples based on different criteria to ensure that the selected samples are representative of the majority class but are closer to the minority class. The specific criteria used by Near Miss include:

1. **NearMiss-1**: This strategy selects samples from the majority class that have the minimum average distance to the k nearest neighbors in the minority class. It focuses on selecting samples that are closest to the minority class instances.

2. **NearMiss-2**: This strategy selects samples from the majority class that have the farthest distance to the k nearest neighbors in the majority class. It aims to select samples that are the farthest away from other majority class instances, reducing redundancy.

3. **NearMiss-3**: This strategy is a variant of NearMiss-2 but selects samples based on the nearest neighbors in both the majority and minority classes. It aims to ensure that the selected samples are not only far away from the majority class but also close to the minority class instances.

By selectively removing samples from the majority class using Near Miss, the algorithm reduces the imbalance between the classes. This can help prevent the classifier from being biased towards the majority class and improve the classification performance on imbalanced datasets.

It's important to note that the choice of under-sampling technique, such as Near Miss, depends on the characteristics of the dataset and the specific problem at hand. Experimentation and evaluation of different techniques are often necessary to determine the most effective approach for a given imbalanced dataset.

# Over Sampling

Over-sampling is a technique used in the field of imbalanced classification to address the issue of imbalanced datasets. It aims to increase the representation of the minority class by artificially generating or duplicating samples.

The goal of over-sampling is to balance the class distribution and provide the classifier with a more balanced training set. By increasing the number of samples in the minority class, over-sampling can help prevent the classifier from being biased towards the majority class and improve its ability to learn patterns and make accurate predictions for the minority class.

There are several common over-sampling techniques:

1. **Random Over-sampling**: This technique randomly duplicates samples from the minority class to match the number of samples in the majority class. It is a simple and straightforward method but may lead to overfitting if not used carefully.

2. **SMOTE (Synthetic Minority Oversampling Technique)**: SMOTE is a popular over-sampling technique that generates synthetic samples for the minority class by interpolating between existing minority class samples. It creates new samples by selecting a sample from the minority class and generating synthetic samples along the line segments connecting it to its nearest neighbors.

3. **ADASYN (Adaptive Synthetic Sampling)**: ADASYN is an extension of SMOTE that adjusts the synthesis of samples based on the density of the minority class. It generates more synthetic samples in regions of the feature space where the minority class is underrepresented.

4. **SMOTE-ENN**: SMOTE-ENN combines the over-sampling technique of SMOTE with the under-sampling technique of Edited Nearest Neighbors (ENN). It first applies SMOTE to generate synthetic samples for the minority class and then uses ENN to remove any potentially misclassified samples from both the majority and minority classes.

It's important to note that over-sampling can introduce some risks, such as overfitting, where the model becomes too specialized to the training data. It is advisable to carefully evaluate the performance of the classifier on a separate validation or test set to ensure that over-sampling has improved the model's generalization ability.

The choice of over-sampling technique depends on the specific characteristics of the dataset and the problem at hand. Experimentation and evaluation of different techniques are often necessary to determine the most effective approach for a given imbalanced dataset.

# XGBoost Classifier

XGBoost (Extreme Gradient Boosting) is a popular and powerful machine learning algorithm for both classification and regression tasks. It is based on gradient boosting, which is an ensemble learning method that combines multiple weak learners (decision trees) to create a strong predictive model.

The XGBoost algorithm has gained widespread popularity and has been successful in various machine learning competitions due to its exceptional performance and scalability. Here are some key features and characteristics of the XGBoost algorithm:

1. **Gradient Boosting**: XGBoost builds an ensemble of weak learners (decision trees) sequentially. Each subsequent tree is trained to correct the mistakes made by the previous trees, focusing on the samples that were misclassified or have high residuals. This iterative process gradually improves the overall predictive performance of the model.

2. **Regularization**: XGBoost includes built-in regularization techniques to prevent overfitting and improve generalization. It uses both L1 (Lasso) and L2 (Ridge) regularization terms to control the complexity of the trees and avoid excessive specialization to the training data.

3. **Optimized Performance**: XGBoost is designed to be highly efficient and scalable. It implements various techniques such as parallelization, column block for sparse data, and cache-aware access to optimize training and prediction speed. These optimizations make XGBoost suitable for handling large datasets and real-world applications.

4. **Feature Importance**: XGBoost provides a feature importance score, which indicates the relative importance of each feature in the prediction task. This information can be useful for feature selection, understanding the underlying patterns, and interpreting the model.

5. **Hyperparameter Tuning**: XGBoost offers a wide range of hyperparameters that can be tuned to optimize the model's performance. These include parameters related to tree structure, regularization, learning rate, subsampling, and more. Proper hyperparameter tuning is essential to achieve the best results with XGBoost.

# Ada Boost

AdaBoost (Adaptive Boosting) is a machine learning algorithm that belongs to the family of ensemble methods. It is primarily used for classification tasks, although it can be extended to regression as well. AdaBoost combines multiple weak classifiers to create a strong predictive model.

Here are the key features and characteristics of AdaBoost:

1. **Boosting Technique**: AdaBoost is a boosting algorithm, which means it sequentially combines weak classifiers to build a strong classifier. The weak classifiers are typically simple models, such as decision trees with a small depth or stumps (decision trees with only one level).

2. **Adaptive Weighting**: AdaBoost assigns weights to each training sample, and these weights are adjusted after each iteration to give more importance to misclassified samples. It focuses on those samples that are difficult to classify correctly and tries to improve the performance on them in subsequent iterations.

3. **Iterative Training**: AdaBoost trains weak classifiers iteratively, with each iteration attempting to correct the mistakes made by the previous weak classifiers. During each iteration, the weak classifier is trained on a modified version of the training set, where the weights of the misclassified samples are increased.

4. **Weighted Voting**: In the final model, the weak classifiers are combined by weighted voting. Each weak classifier contributes to the final prediction based on its performance and the weight assigned to it during training. The weights reflect the classifier's accuracy, with more accurate classifiers having higher weights.

5. **Model Adaptability**: AdaBoost is adaptable to different types of weak classifiers and can work with any learning algorithm that provides weak learners. It can be combined with various base classifiers, such as decision trees, support vector machines (SVMs), or neural networks.

# Random Forest

Random Forest is a popular machine learning algorithm that belongs to the ensemble learning methods. It is primarily used for both classification and regression tasks. Random Forest combines multiple decision trees to create a powerful predictive model.

Here are the key features and characteristics of Random Forest:

1. **Ensemble of Decision Trees**: Random Forest builds an ensemble of decision trees, where each tree is trained independently on a random subset of the training data. This process is known as bagging (bootstrap aggregating). By combining the predictions of multiple trees, Random Forest can make more accurate and robust predictions.

2. **Random Feature Selection**: During the construction of each decision tree in the Random Forest, a random subset of features is selected as candidates for splitting at each tree node. This random feature selection helps to decorrelate the trees and reduce the variance of the model, improving the overall performance.

3. **Bootstrap Aggregation**: Random Forest creates each decision tree using a bootstrap sample of the training data. This means that each tree is trained on a random subset of the training examples, allowing for diversity and reducing the impact of outliers or noisy data.

4. **Voting or Averaging**: In classification tasks, Random Forest combines the predictions of individual decision trees using majority voting, where the class with the most votes is selected as the final prediction. In regression tasks, the individual tree predictions are averaged to obtain the final prediction.

5. **Feature Importance**: Random Forest provides a measure of feature importance, which indicates the relative importance of each feature in the prediction task. This information can be used for feature selection, understanding the impact of different features, and interpreting the model.

Random Forest is known for its ability to handle high-dimensional data, deal with missing values and outliers, and provide good generalization performance. It is robust against overfitting and tends to have lower variance compared to individual decision trees.






