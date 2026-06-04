# 🧠 CNN Image Classification on CIFAR-10

A Convolutional Neural Network (CNN) built from scratch using Keras and TensorFlow to classify images from the CIFAR-10 dataset into 10 categories.

---

## 📌 Project Overview

This project demonstrates how CNNs overcome the limitations of traditional fully connected neural networks for image classification — specifically the loss of spatial features and the explosion of parameters. The model is trained on the CIFAR-10 dataset and can classify images into one of 10 object categories.

---

## 🗂️ Dataset — CIFAR-10

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of:
- **60,000** colour images of size **32×32 pixels**
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **50,000 training** images and **10,000 test** images

The dataset is loaded directly via `keras.datasets.cifar10` — no manual download needed.

---

## 🏗️ Model Architecture

```
Input (32x32x3)
    │
Conv2D (16 filters, 2x2, ReLU, same padding)
MaxPooling2D (2x2)
    │
Conv2D (32 filters, 2x2, ReLU, same padding)
MaxPooling2D (2x2)
    │
Conv2D (64 filters, 2x2, ReLU, same padding)
MaxPooling2D (2x2)
Dropout (0.3)
    │
Flatten
Dense (500, ReLU)
Dropout (0.4)
    │
Dense (10, Softmax)  ← Output: 10 class probabilities
```

- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: RMSprop  
- **Epochs**: 15  
- **Batch Size**: 32  

---

## 📁 Project Structure

```
cnn-cifar10-classification/
│
├── CNN_Python_Implementation.ipynb   # Main Jupyter notebook (theory + code)
├── cnn_cifar10.py                    # Clean standalone Python script
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files to exclude from Git
└── README.md                         # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cnn-cifar10-classification.git
cd cnn-cifar10-classification
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook CNN_Python_Implementation.ipynb
```

Or run the Python script directly:
```bash
python cnn_cifar10.py
```

---

## 🚀 How It Works

1. **Load Data** — CIFAR-10 is loaded and split into train, validation, and test sets
2. **Preprocess** — Pixel values normalised to [0, 1]; labels one-hot encoded
3. **Build Model** — 3 Conv+Pool blocks followed by Dense layers with Dropout regularisation
4. **Train** — ModelCheckpoint saves the best weights based on validation accuracy
5. **Evaluate** — Final accuracy measured on the unseen test set
6. **Predict** — Model predicts the class of any new 32×32 image

---

## 📊 Results

| Metric              | Value         |
|---------------------|---------------|
| Training Epochs     | 15            |
| Batch Size          | 32            |
| Validation Split    | 5,000 images  |
| Test Set Size       | 10,000 images |

> Model weights are saved automatically to `model.weights.best.keras` during training.

---

## 🧪 Key Concepts Covered

- Why CNNs outperform MLPs for image data (spatial invariance, parameter efficiency)
- Convolution, padding, and stride operations
- Pooling layers and feature aggregation
- Dropout for regularisation
- One-hot encoding and categorical crossentropy

---

## 🛠️ Tech Stack

| Tool         | Purpose                    |
|--------------|----------------------------|
| Python 3.x   | Core language              |
| TensorFlow   | Deep learning backend      |
| Keras        | High-level model API       |
| NumPy        | Numerical operations       |
| Matplotlib   | Visualisation              |
| OpenCV (cv2) | Image loading              |

---

## 👩‍💻 Author

**Sasikala Madhu**  
Data Analyst | Aspiring ML Engineer  
📧 sasivetri2025@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
