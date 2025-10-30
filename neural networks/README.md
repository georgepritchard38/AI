## Neural Networks

This project builds, trains, and evaluates two neural networks using PyTorch, one deeper and one shallower model in the "neural_net.py" program. The dataset used to train and test both models is the FashionMNIST torchvision embedded dataset of articles of clothing, with 10 discrete clothing type labels. Both models output their training set accuracy and loss status with each training epoch when "train_model" is called, as well as their total test loss and accuracy when "evaluate_model" is called. The "predict_labels" function returns the top three most likely labels for a given image on an already trained model. I have included sample_output.txt which is the output of "neural_net.py" with all of the current hyperparameters seen in "neural_net.py" unchanged. The program trains and tests the shallow model, then the deeper model, and then predicts the label of the first loaded image using the deeper model. Note in "sample_output.txt" that the hyperparameters are tuned to favour the shallow model, which performs better than the deep model in this case.

## provided by instructors:

neural_net.py structure and headers

## personal contributions:

implementation of neural_net.py

sample_output.txt
