# AutoPCaSeg
This repository contains the code of the AutoPCaSeg application, developed for my Bachelor Thesis. It is a deep learning-based software for automatic segmentation of prostate cancer lesions in T2-weighted MRI. It is structured as follows.

The **utils** folder contains all the necessary utilities:

		* preprocess.py -> contains all the necessary functions to preprocess the T2W images and their associated segmentation masks. 

		* postproccess.py -> functions to postprocess the segmentation masks predicted by the models.

		* metrics.py -> implementation of the evaluation metrics 

		* losses.py -> contains the utilities needed to compute the loss functions.

		* train.py -> includes all the functionality to train the models, evaluate them and make predictions on new data with them. 

		* cross_val.py -> implementation of cross-validation used to evaluate the models.

		* save.py -> functions to save the results of the models. 

The file *run_cv.py* executes 3-fold cross-validation with different parameters of the models and the training process, while the file *run_test.py* is intended to test the models on different test sets. 

This software is written in Python 3.8.10 and has the following dependencies:
		+ Pytorch 1.10.2
		+ Monai [https://monai.io/](https://monai.io/)
		+ Numpy 1.21.2 
		+ Matplotlib 3.5.1
		+ Scikit-learn 1.0.2
		+ Segmentation-models-pytorch 0.3.0 [https://github.com/chsasank/segmentation_models.pytorch](https://github.com/chsasank/segmentation_models.pytorch)
		+ Scikit-image 0.19.2