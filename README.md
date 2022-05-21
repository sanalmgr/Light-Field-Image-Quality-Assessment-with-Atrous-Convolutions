# Light-Field-Image-Quality-Assessment-with-Atrous-Convolutions
In this work, we propose a novel no-reference LF-IQA method that takes into account both LF angular and spatial information. The proposed method is composed of two processing streams with identical blocks of Convolutional Neural Network (CNN), Atrous Convolution layers (ACL), and a regression block for quality prediction. 

## Code:
## Training Model:
1. Prepare the horizontal and vertical EPIs using the method MultiEPL https://bit.ly/3Da8fB6.
2. To train the model, import functions from train_model.py file, and pass the parameters accordingly.
