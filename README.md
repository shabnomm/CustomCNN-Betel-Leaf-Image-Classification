# CustomCNN-Betel-Leaf-Image-Classification

Group Members:
1. Shanzia Shabnom Mithun
2. Hiya Saha

Dataset:
Dataset Name: Betel Leaf Image Dataset from Bangladesh

Source: Mendeley Data
This dataset contains images of betel leaves classified into two categories (Healthy & Diseased).

Project Overview:
We designed and trained a novel Convolutional Neural Network (CNN) from scratch for classifying betel leaf images.
The architecture is meaningfully different from the sample models shared on Google Classroom.
Our pipeline includes:
  -Custom CNN design
  -Data augmentation
  -Mixed precision training (torch.cuda.amp)
  -Early stopping & LR scheduling
  -Evaluation metrics (Precision, Recall, F1)
  -Confusion Matrix & Loss Curve plots

The notebook is divided into clear sections:

  -Data Loading & Preprocessing
  -Model Architecture
  -Training
  -Evaluation & Visualization
  -Model Weights:

Best model weights (custom_cnn_best.pt) are saved based on highest validation F1 score during training.

Reproduction Steps:
To reproduce the results:
1️. Open the notebook CustomCNN_CSE366.ipynb in Google Colab or Kaggle.
2️. Make sure the dataset is available at:
/kaggle/input/betel-leaf-image-dataset-from-bangladesh/Betel Leaf Image Dataset from Bangladesh
(or change the data_dir path in the notebook if needed)
3️. Run all cells in order.
4️. Training and evaluation metrics will be printed, along with loss curves and confusion matrix.

Notes:
This code is written in PyTorch.

Uses mixed precision (AMP) for faster training on GPU.
Early stopping & learning rate scheduler are enabled.
Fully reproducible with set_seed() function.
