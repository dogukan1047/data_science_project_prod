
# Used the script in the model_two_prod folder

# 1. Job Description:
This document outlines the steps involved in creating and enhancing a machine learning model designed to classify images accurately. The main method used is called a Multi-Layer Perceptron (MLP) classifier. The ultimate aim is to develop a model capable of correctly identifying the content of images, which is crucial for various applications like image recognition, medical diagnosis, and more. The process entails several stages, including data preparation, model selection, parameter tuning, training, evaluation, and result visualization
# 1.1.Architectural Diagram:
The architectural framework intricately maps out the constituent components and the systematic workflow of the machine learning pipeline. It encompasses a meticulous orchestration of various stages, commencing with the rigorous prelude of data preprocessing. This involves meticulous steps such as ingesting data from a Comma-Separated Values (CSV) file, partitioning the dataset into discerning subsets for training and evaluation purposes, and meticulously transforming image vectors into discernible numerical arrays. The crux of the architectural design pivots on the meticulous configuration of the MLP classifier. This entails a judicious calibration of a plethora of hyperparameters, including but not limited to, the sizes of the hidden layers, the activation functions governing the neurons' behavior, the regularization parameters dictating the network's robustness, and the maximum iteration counts, delineating the trainingepochs. 
	
![image](https://github.com/dogukan1047/data_science_project_prod/assets/70372233/86fd8f45-1543-4b19-abd5-bb94c0a00a20)

## split_images.py: 
This code defines a function called split_image that takes an image file path along with parameters for the number of rows and columns to divide the image into. It splits the image into smaller pieces based on the specified rows and columns, and optionally squares each piece to ensure uniformity. The resulting pieces are saved as separate image files in the specified output directory. Additionally, the function provides options to square the pieces and perform cleanup operations.
![image](https://github.com/dogukan1047/data_science_project_prod/assets/70372233/e2957ada-887c-489b-be9c-96c9ee595354)

## csv_creater.py:
This script converts images from a directory into vectors and appends them, along with corresponding labels, to a CSV file. It resizes the images to 28x28 pixels, converts them to grayscale, flattens them into vectors, and assigns labels based on their order. Finally, it writes the image vectors and labels to the CSV file.
	
## Model_script.py:
This code trains a Multi-Layer Perceptron (MLP) classifier:
1.It reads data from a CSV file.
2.Splits the data into input (X) and output (y) variables.
3.Uses GridSearchCV to find the best model and hyperparameters.
4.Evaluates the best model's performance on a test set.
5.Saves the best model to a file.
6.Visualizes loss and accuracy during training.
7.Displays final training and test accuracies

## Prediction.py:
This code takes an image vector as input from the user, uses a pre-trained model to  make a prediction, and then prints the prediction result.
# 2.Implementation of Work
The journey begins with the extraction of data from a CSV file, containing information about images and their corresponding labels. This data is then divided into two sets: one for training the model and the other for testing its performance. The model selection process involves choosing the best configuration for the MLP classifier, which includes deciding on the number of hidden layers, activation functions, regularization parameters, and the maximum number of iterations. Grid search, a technique for hyperparameter optimization, is employed to find the optimal settings for the model. Once the best configuration is identified, the model is trained using the training data to learn patterns and relationships between images and their labels
# 3.Operation of Work:
Throughout the training process, close attention is paid to the model's performance, particularly its ability to minimize errors and improve accuracy over time. This is monitored by tracking the loss values, which indicate how well the model is fitting the training data. Additionally, the model's accuracy is evaluated using the test data to assess its generalization ability. The evolution of the model's performance is visualized through graphs, illustrating the changes in loss values and accuracy scores over iterations. These visualizations provide valuable insights into the effectiveness of the model and highlight areas for potential improvement.
# 4.Conclusion:
In conclusion, the development of the machine learning model represents a significant achievement in the field of image classification. By leveraging the power of MLP classifiers and employing rigorous optimization techniques, we've been able to create a model that exhibits promising performance in identifying image content accurately. The iterative process of training, evaluation, and refinement has led to a better understanding of the data and the underlying patterns within it. Moving forward, continued efforts to enhance the model's performance and scalability will be essential for its successful deployment in real-world applications

 
![image](https://github.com/dogukan1047/data_science_project_prod/assets/70372233/a9e2a859-1c53-490e-a78a-0b986bfa744a)


 
![image](https://github.com/dogukan1047/data_science_project_prod/assets/70372233/5f267b4e-4f57-4e02-9f7c-c520c3550a35)

Bibliography:
*  Our Documentation | Python.org
*  sklearn.neural_network.MLPClassifier — scikit-learn 1.4.2 documentation
*  Developer guides (keras.io)
*  https://pandas.pydata.org/docs/


