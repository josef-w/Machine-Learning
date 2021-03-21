# Machine Learning with numpy
Implement classic machine learning algorithms using numpy(without use of sklearn). It will be a good place to start with if you want to understand how they work. The code are adjusted from the assignments of machine learning courses in UBC. The dict of the files and the modulars/functions that contained in each file are as follows.


    └─code
        ├─findMin.py
        │  ├─findMin
        │  └─findMinL1
        ├─kmeans.py
        │  └─Kmeans
        ├─knn.py
        │  └─KNN
        ├─least_squared_regression.py
        │  ├─LeastSquares
        │  ├─WeightedLeastSquares
        │  ├─LinearModelGradient
        │  └─LeastSquaresBias
        ├─linear_autoregressive.py
        │  └─LeastSquaresAutoregressive
        ├─logistic_regression.py
        │  ├─logReg
        │  ├─logRegL1
        │  ├─logRegL2
        │  ├─logRegL0
        │  └─kernelLogRegL2
        ├─linear_classifier.py
        │  ├─leastSquaresClassifier
        │  ├─logLinearClassifier
        │  └─softmaxClassifier
        ├─decision_stump.py
        │  ├─DecisionStumpErrorRate
        │  ├─DecisionStumpEquality
        │  ├─DecisionStumpInfoGain
        │  └─DecisionStumpGiniIndex
        ├─decision_tree.py
        │  ├─DecisionTree
        ├─random_stump.py
        │  ├─RandomStumpInfoGain
        │  └─RandomStumpGiniIndex
        ├─random_tree.py
        │  ├─RandomTree
        ├─random_forest.py
        │  ├─RandomForest
        ├─naive_bayes.py
        │  ├─NaiveBayes
        ├─pca.py
        │  ├─PCA
        │  ├─AlternativePCA
        │  ├─misc
        │  └─RobustPCA
        ├─neural_net.py
        │  ├─NeuralNet
        ├─main.py
        
The structure of the functions and their protocol are highly unified. So generally, you can simplely fit a model and perform prediction by 

      model=<name of the function>
      model.fit(X,y)#For unsupervised learning, it will be model.fit(X). Some model like knn, decision tree need hyper parameter.
      y_pre=model.predict(Xtest)

`main.py` contains sample code to run some of the machine learning algorithms. If you run `python main.py -q 1`, it will load a dataset, and fit a decision tree with depth of 6, and evaluate training and test error, and finally plot the decision boundary. The dataset containing longitude and latitude data for 400 cities in the US, along with a class label indicating whether they were a “red” state or a “blue” state in the 2012.The first column of X contains the longitude and the second contains the latitude, while the variable y is set to 0 for blue states and 1 for red states
election(http://simplemaps.com/static/demos/resources/us-cities/cities.csv). The result is as follows.

![image](https://github.com/josef-w/Machine-Learning/blob/main/figs/decisionBoundary.png)

You can also change your model into decision stump,random tree, random forest, or change stump class, and compare the results. For example, if you fit a random tree with depth of 6, you may get a result like this

    RandomTree:max_depth=6
    Training error: 0.115
    Testing error: 0.172
    Time:0.102


    DecisionTree:stump_class=DecisionStumpGiniIndex,max_depth=6
    Training error: 0.018
    Testing error: 0.092
    Time:0.208
    
This implies that random tree could greatly reduce the runnning time, though sacrificing some performance slightly.

If you run `python main.py -q 2`, it will load a dataset same with q1, and fit a knn with k=4, and evaluate training and test error, and finally plot the decision boundary. The result is as follows.

![image](https://github.com/josef-w/Machine-Learning/blob/main/figs/knn.png)

If you run `python main.py -q 3` it will load a dataset with two features, and fit 50 Kmeans(k=4) with different initalizations, and choose a best one with min SSE score, then evaluate and plot.The result is as follows.

![image](https://github.com/josef-w/Machine-Learning/blob/main/figs/kmeans_basic.png)

If you run `python main.py -q 4` it will fit a robust linear regression(the loss function is represented by L1 norm). The result is as follows. It can be seen that the regression is less sensitive to the outliers.

![image](https://github.com/josef-w/Machine-Learning/blob/main/figs/least_squares_robust.png)

If you run `python main.py -q 5` it will load a dataset X where each row contains the pixels from a single frame of a video of a highway, and fit a PCA (solved by gradient descend), and then fit a robust PCA(using L1 norm). The goal is to remove the cars from the background. The result comparing these two models is as follows. It can be seen that robust PCA remove the cars better than PCA.

![image](https://github.com/josef-w/Machine-Learning/blob/main/figs/pca_highway.png)
