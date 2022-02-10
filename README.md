# Light-Field-Image-Quality-Assessment-with-Atrous-Convolutions
Epipolar plane image (EPI) is a 2D representation of a Light Field image (LFI). It describes the shift of the pixel information over the angular axis. In case of discontinuity in the angular domain, lines on the EPI can become distorted. To explore dense features of EPIs, we propose a novel no-reference (NR) Light Field Image Quality Assessment (LF-IQA) method called CNN-ACL that employs a Convolutional Neural Network (CNN) with Atrous Convolution layers (ACL). The proposed LF-IQA method processes the horizontal and vertical EPIs in the format of two streams. Both streams have identical blocks of CNN and ACL that pass the processed information to the regression block for quality prediction. Results show that CNN-ACL method outperforms state-of-the-art methods.

## Code:
## Training Model:
1. Prepare the horizontal and vertical EPIs using the code Coming Soon.
2. To train the model, import functions from training_model.py file, and pass the parameters accordingly.
