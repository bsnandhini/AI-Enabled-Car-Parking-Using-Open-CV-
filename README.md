# AI Enabled Car Parking – Python & OpenCV

## Project Overview

AI Enabled Car Parking is a computer vision and machine learning project designed to detect whether parking spaces are **Free** or **Full** from parking-lot images.

The project uses Python, OpenCV, and machine learning/deep learning techniques to classify parking-space images and evaluate model performance.

## Objectives

* Detect the availability of parking spaces from images.
* Classify parking spaces as **Free** or **Full**.
* Apply image preprocessing and machine learning techniques.
* Train and evaluate different classification models.
* Visualize model performance using evaluation metrics.

## Technologies Used

* **Python**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**
* **Git & GitHub**

## Models

The project includes different machine learning and deep learning approaches:

* Logistic Regression
* Multi-Layer Perceptron (MLP)
* Convolutional Neural Network (CNN)

## Project Structure

```text
AI Enabled Car Parking – Python & OpenCV/
│
├── dataset/
│   ├── Free/
│   └── Full/
│
├── documentation/
│   └── Project documentation files
│
├── output_images/
│   ├── confusion_matrix.png
│   ├── model_accuracy.png
│   ├── model_loss.png
│   ├── logistic_regression_cost.png
│   ├── properly_classified_images.png
│   └── misclassified_images.png
│
├── src/
│   ├── data_loader.py
│   ├── logistic_regression.py
│   ├── mlp_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
│
├── .gitignore
└── README.md
```

## Project Workflow

```text
Parking Space Images
        ↓
Data Loading
        ↓
Image Preprocessing
        ↓
Feature Preparation
        ↓
Model Training
        ↓
Model Prediction
        ↓
Model Evaluation
        ↓
Visualization of Results
```

## Evaluation

The trained models are evaluated using classification and visualization techniques such as:

### model Accuracy 
     output_images\model_accuracy.png
### model loss 
    output_images\model_loss.png
###  Confusion Matrix 
     output_images\confusion_matrix.png
### Training/validation performance 
    output_images\dataset_sample.png
### Properly classified images 
    output_images\properly_classified_images.png
### Misclassified images 
    output_images\misclassified_images.png
### Logistic Regression cost 
    output_images\logistic_regression_cost.png


The generated visualizations are available in the `output_images` directory.

##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/bsnandhini/AI-Enabled-Car-Parking-Using-Open-CV-.git
```

### 2. Navigate to the project

```bash
cd AI-Enabled-Car-Parking-Using-Open-CV-
```

### 3. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install required packages

```bash
pip install -r requirements.txt
```

### 5. Run the required Python scripts

For example:

```bash
python src/logistic_regression.py
```

```bash
python src/mlp_model.py
```

```bash
python src/evaluate.py
```

## Dataset

The project uses parking-space images categorized into two classes:

* **Free**
* **Full**

The dataset is used for training and evaluating the classification models.

## Note About Trained Models

Large trained model files such as `.keras` and `.h5` are excluded from this GitHub repository using `.gitignore` because of their large file size.

The source code and project documentation are included in the repository.

## Future Enhancements

* Real-time parking-space detection using a live camera feed.
* Integration with a complete parking management system.
* Web-based dashboard for displaying parking availability.
* Automatic parking-space counting.
* Deployment as a real-time computer vision application.
