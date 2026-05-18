# Kidney Stone Detection using CNN and Harris Hawks Optimisation

**Course:** Advanced Machine Learning — GUtech  
**Date:** December 2025  
**Accuracy:** 93.51% (Baseline CNN) | 92.82% (CNN-HHO Hybrid)

###### **Project Overview**

This project implements a hybrid deep learning system for kidney stone detection from axial CT images. A Convolutional Neural Network (CNN) is used as the baseline classifier, and CNN + Harris Hawks Optimization (HHO) is applied to optimize key hyperparameters such as learning rate, dropout, batch size, and image resolution.

The model was trained and evaluated using the Kidney Stone Axial CT Imaging Colorized Dataset from Kaggle, with the data split into training, validation, and test sets. The goal is to study whether a bio-inspired optimization method can improve model performance under limited computational resources, while maintaining clinical relevance for medical imaging tasks.

###### 

###### **Prerequisites**

Python 3.11

macOS, Windows, and Linux 

CPU hardware supported

sufficient disk storage, As the dataset is relatively large



###### **Environment Setup**

create and activate the virtual environment:


***python3 -m venv .venv***

***source .venv/bin/activate   for macOS/Linux***

***.venv\\Scripts\\activate    for Windows***


All required dependencies are installed using the provided requirement file:

***pip install -r requirements.txt***


###### **Dataset Instructions**

Download the Kidney Stone Axial CT Imaging Colorized Dataset from:



***https://www.kaggle.com/datasets/shuvokumarbasakbd/kidney-stone-axial-ct-imaging-colorized-dataset***



Place the dataset into class-specific directories corresponding to stone and non-stone cases.
The dataset must be placed in the following structure:

data/raw/axial_ct_kidney/
├── Stone/
└── Non-Stone/


**Dataset Splitting**

Split the dataset into training, validation, and test sets by running:

**python src/split\_dataset.py data/raw/axial\_ct\_kidney data/splits** 


###### **Running the Baseline CNN**

The baseline CNN model is trained using:

***python src/train.py***

Outputs are automatically saved to:

***runs/baseline/***


###### **Running CNN with Harris Hawks Optimization (HHO)**



For the optimized approach, Harris Hawks Optimization is applied to search for improved CNN hyperparameter configurations. 

The optimization and final training process can be executed using:



***python src/hho_train.py***



Outputs are saved to:

***runs/hho/***


###### **Evaluation Metrics**

Model performance is evaluated using:

accuracy

precision

recall

F1-score

The F1-score is emphasized, as it provides a balanced assessment of false positives and false negatives, which is particularly important in medical diagnosis applications.


###### **Hardware and Runtime Notes**

All experiments were conducted on CPU-only hardware. The HHO optimization stage is computationally expensive and may require extended runtime on non-GPU systems.

###### 
