# Multi-Module Hybrid Deep Learning System for Digital Image Forgery Detection

##  Overview

This project presents a **hybrid deep learning-based system** designed to detect whether a digital image is **authentic or forged**. The system integrates multiple techniques such as **Convolutional Neural Networks (CNNs), Error Level Analysis (ELA), and attention mechanisms** to improve detection accuracy and reliability.

It not only classifies images but also **localizes manipulated regions using heatmaps**, making the system more interpretable and useful in real-world applications.

---

##  Problem Statement

With the rapid growth of image editing tools and AI-based manipulation techniques, detecting forged images has become extremely difficult. Traditional methods fail to detect complex forgeries and cannot identify manipulated regions.

This project aims to:

* Detect forged images accurately
* Identify manipulated regions
* Handle advanced and realistic image manipulations

---

##  Key Features

*  Detects **Authentic vs Forged images**
*  Uses **Hybrid Deep Learning Architecture**
*  Supports **Error Level Analysis (ELA)**
*  Provides **Heatmap Visualization (Grad-CAM)**
*  Real-time analysis using **Streamlit Web App**
*  Displays performance metrics (Accuracy, Precision, Recall)

---

##  System Architecture

The system follows a multi-stage pipeline:

1. Image Upload
2. Preprocessing (Resizing, Normalization, ELA)
3. Feature Extraction using CNN
4. Model Prediction
5. Heatmap Generation (Grad-CAM)
6. Result Display

---

##  Models Used

* Custom Hybrid CNN
* CNN_D64
* CNN_D64_128
* CNN_D64_128_128 (Best Performing Model)
* VGG16
* EfficientNet

 Best Model Performance:

* Accuracy: **91.57%**
* Precision: **88.95%**
* Recall: **91.26%**
* AUC: **0.9611**

---

##  Project Structure

```
Digital Image Forgery Detection/
│── app.py                # Main application
│── config.json           # Configuration file
│── metrics.csv           # Evaluation results
│── html.html             # Frontend interface
│── *.keras               # Trained models
```

---

##  Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Streamlit

---

##  System Requirements

**Software:**

* Windows 10/11
* Python

**Hardware:**

* Processor: Intel i5 / Ryzen
* RAM: 8GB

---

##  How to Run the Project

1. Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
```

2. Navigate to project folder:

```
cd Digital Image Forgery Detection
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the application:

```
python app.py
```

---

##  Applications

*  Journalism (Fake image detection)
*  Legal & Forensics
*  Cybersecurity
*  Medical Imaging Verification
*  Social Media Content Validation

---

##  Future Scope

* Improve detection using advanced transformer models
* Support video forgery detection
* Deploy on cloud for large-scale usage
* Improve localization accuracy

---

##  Authors

* Harika Chamarthi
* Navya Sri
* Srujana
* Siva Sahithi

---

##  License

This project is developed for academic purposes.
