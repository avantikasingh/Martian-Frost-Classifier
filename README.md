# Project Title: Martian Frost Classifier

## Overview

This project focuses on developing a classifier to identify frost in Martian terrain using state-of-the-art deep learning techniques. The dataset used for this project is derived from high-resolution images captured by the HiRISE (High-Resolution Imaging Science Experiment) camera on NASA's Mars Reconnaissance Orbiter. The goal is to contribute to the study of Mars' seasonal frost cycle and its implications for the planet's climate and surface evolution over the past 2 billion years.

## Dataset

Dataset used: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=doi:10.48577/jpl.QJ9PYA
This dataset consists of individual tiles, each representing a 299x299 pixel crop from annotated HiRISE subframes. The images are labeled as either 'frost' or 'background' based on human annotations. The dataset is organized into 'background' and 'frost' classes, and it contains a total of 119,920 tiles from 214 subframes.

## Training CNN + MLP

The initial approach involves training a three-layer Convolutional Neural Network (CNN) followed by a dense layer. To enhance model generalization, the training set undergoes empirical regularization through image augmentation techniques such as cropping, random zooming, rotation, flipping, contrast adjustment, and translation using OpenCV. ReLU activation functions are used, along with softmax, batch normalization, a dropout rate of 30%, L2 regularization, and the ADAM optimizer. The model is trained for at least 20 epochs with early stopping based on validation set performance. Precision, recall, and F1 score are reported.

## Transfer Learning

Recognizing the challenges of small image datasets, transfer learning is employed using pre-trained models (EfficientNetB0, ResNet50, and VGG16). The last fully connected layer is trained, while all previous layers are frozen. Image augmentation and regularization techniques are applied similarly to the CNN + MLP model. The model is trained for at least 10 epochs, and the results are compared with the CNN + MLP model. Precision, recall, and F1 score are reported, and the outcomes are discussed.

## Repository Contents

- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, and model training.
- `test_source_images.txt/`: test data folders
- `train_source_images.txt/`: train data folders
- `val_source_images.txt/`: validation data folders
- `README.md`: Comprehensive documentation of the project.

## Model Performance Summary

- `CNN Model:`
   >Precision: 0.76
   Recall: 0.76
   F1-score: 0.75
   Accuracy: 0.76

- `EfficientNetB0 Transfer Learning:`

    >Precision: 0.57
    Recall: 0.65
    F1-score: 0.53
    Accuracy: 0.65

- `ResNet50 Transfer Learning:`

    >Precision: 0.43
    Recall: 0.66
    F1-score: 0.52
    Accuracy: 0.66

- `VGG16 Transfer Learning:`

    >Precision: 0.52
    Recall: 0.50
    F1-score: 0.51
    Accuracy: 0.50


The CNN model performs the best overall with the highest precision, recall, and F1-score, resulting in the highest accuracy. The EfficientNetB0 Transfer Learning model has a lower performance compared to the CNN model, and ResNet50 Transfer Learning and VGG16 Transfer Learning models have the lowest accuracy among the models listed.



## Conclusion

This project leverages real data from NASA JPL to build a robust classifier for identifying frost in Martian terrain. By combining CNN + MLP and transfer learning approaches, the project explores the best methods for handling limited image datasets and provides insights into the effectiveness of different deep learning architectures for this specific task.
