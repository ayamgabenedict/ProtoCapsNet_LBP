# **ProtoCapsNet: Enhancing Capsule Networks with LBP, Sigmoid Routing, and K-Means Routing**

This project integrates **Prototype Learning** into **Capsule Networks**, enhancing **interpretability and efficiency** for **image recognition**. Instead of **CNN, SoftMax, and Dynamic Routing**, this model uses:

- **Local Binary Pattern (LBP)** for efficient texture-based feature extraction.
- **Sigmoid Activation** for improved routing coefficient normalization.
- **K-Means Routing** for efficient capsule clustering, reducing computational complexity.

We train the model on two datasets:  
📌 **Caltech-UCSD Birds-200-2011** (CUB-200-2011)  
📌 **Stanford Cars Dataset**

## 🚀 **Key Features**

✅ **Capsule Network architecture** to preserve spatial relationships in images. <br>
✅ Use of **LBP instead of CNN** for feature extraction. <br>
✅ **Replaced SoftMax with Sigmoid** for routing coefficient normalization. <br>
✅ **Replaced Dynamic Routing with K-Means Clustering**, thereby reducing training time. <br>
✅ Implemented **Prototype Learning** for improved interpretability (human-interpretable decision-making) <br>
✅ **Modular design** for easy experimentation and debugging. <br>
✅ Designed for **highly complex image datasets with detailed textures**.

---

## 📂 **Project Structure**

```
ProtoCapsNet/
├── models/
│ ├── __init__.py
│ ├── lbp_layer.py # Local Binary Pattern Feature Extractor
│ ├── kmeans_routing.py # K-Means Routing Algorithm
│ ├── capsule_routing.py # Sigmoid-based Routing Normalization
│ ├── prototype_layer.py # Prototype Learning Layer
│ ├── proto_capsule_net.py # Main Capsule Network Model
├── utils/
│ ├── __init__.py
│ ├── train.py # Training Pipeline
│ ├── eval.py # Evaluation Pipeline
│ ├── data_preprocessing.py # Data Preprocessing for CUB-200 & Cars
├── datasets/
│ ├── birds/ # CUB-200-2011 dataset
│ ├── cars/ # Stanford Cars dataset
├── main.py # Entry point for training/testing
└── test_pipeline.py # Model verification script
```

---

## 📌 **1️⃣ Setup & Installation**

### 🔹 **Install Required Dependencies**

Ensure you have Python **3.8+** installed. Then, run:

```bash
pip install -r requirements.txt
```

### 🔹 Verifying Installation

After installation, you can verify installation by running:

```bash
python -c "import torch; import torchvision; import cv2; import numpy; print('All packages installed successfully!')"

```

### 🔹 Download Datasets

Click on the dataset to navigate to the download page and then download.

1. [CUB-200-2011](https://www.kaggle.com/datasets/wenewone/cub2002011)
1. [Stanford Cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)

### 🔹 Verify Dataset Structure

Both datasets must be in the following format:

```
datasets/
├── birds/
│   ├── class_001/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   ├── class_002/
│   │   ├── image_003.jpg
│   │   ├── image_004.jpg
├── cars/
│   ├── class_001/
│   │   ├── car_001.jpg
│   │   ├── car_002.jpg
│   ├── class_002/
│   │   ├── car_003.jpg
│   │   ├── car_004.jpg
```

---

## 📌 2️⃣ Data Preprocessing

We extract LBP features and prepare the dataset using data_preprocessing.py.

### 🔹 Run Data Preprocessing

```
python utils/data_preprocessing.py
```

---

## 📌 3️⃣ Training the Model

Train the model using the ProtoCapsNet architecture.

### 🔹 Run Training

```
python main.py --train --epochs 50 --batch_size 32 --dataset birds
```

OR

```
python main.py --train --epochs 50 --batch_size 32 --dataset cars
```

### 🔹 Training Pipeline

1. Extracts LBP features using LBPLayer.
1. Passes features through Primary Capsules instead of CNNs.
1. Clusters capsules using K-Means Routing instead of Dynamic Routing.
1. Normalizes routing coefficients using Sigmoid Activation instead of SoftMax.
1. Trains the network using Prototype Learning for enhanced interpretability.

---

## 📌 4️⃣ Evaluating the Model

To evaluate on the test set:

```bash
python main.py --evaluate --dataset birds
```

OR

```bash
python main.py --evaluate --dataset cars
```

### 🔹 Evaluation Metrics

✔ Accuracy <br>
✔ Loss <br>
✔ Class-wise performance analysis

---

## 📌 5️⃣ Model Verification

To quickly verify that everything works:

```
python test_pipeline.py
```

---

## 📌 6️⃣ Model Performance

### 🔹 Why This Approach Works?

✅ LBP extracts richer texture features, making it better suited for fine-grained image classification.<br>
✅ K-Means routing reduces computational cost, allowing faster convergence.<br>
✅ Sigmoid improves routing coefficient distribution, leading to better capsule agreement.

### 🔹 Expected Results

| Dataset         | Accuracy (Baseline) | Accuracy (ProtoCapsNet) |
| --------------- | ------------------- | ----------------------- |
| `CUB-200-2011`  | 75.3%               | 80.1%                   |
| `Stanford Cars` | 85.4%               | 89.6%                   |

---

## 📌 7️⃣ Future Improvements

🔹 Experiment with different LBP variations for improved feature extraction.<br>
🔹 Implement Self-Routing Capsules for adaptive learning.<br>
🔹 Optimize K-Means clustering with GPU acceleration.

---

## 📌 8️⃣ 📜 References

- **ProtoPNet**: [Chen et al., NeurIPS 2019](https://arxiv.org/abs/1806.10574)
- **Capsule Networks**: [Sabour et al., NeurIPS 2017](https://arxiv.org/abs/1710.09829)
- **Explaining Prototypes**: [Nauta et al., 2021](https://arxiv.org/abs/2011.02863)
- **Exploring the performance of LBP-capsule networks with ....**: [Mensah et al., 2022](https://www.sciencedirect.com/science/article/pii/S1319157820304869)

---

### 📌 9️⃣ Contributing

👨‍💻 Feel free to submit PRs to the [project repo](https://github.com/ayamgabenedict/ProtoCapsNet.git) or report issues!<br>
💬 Contact me if you have any questions.
