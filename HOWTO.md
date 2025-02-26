## 📌 Step-by-Step Approach to Building & Running ProtoCapsNet

This guide provides a clear sequence of steps from dataset preparation to making predictions.

---

### 1️⃣ Install Dependencies

First, ensure you have all required packages installed.

```bash
pip install -r requirements.txt
```

### 2️⃣ Download and Extract the Datasets

1. Visit [CUB-200-2011](https://www.kaggle.com/datasets/wenewone/cub2002011) and [Stanford Cars datasets](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) respectively to download the two required datasets.
1. Extract the datasets into the `datasets` directory.

---

### 3️⃣ Preprocess the Data (Apply LBP, Normalize, Split into Train/Test)

Run the preprocessing script to prepare the datasets.

```bash
python utils/data_preprocessing.py
```

✅ This will:

- Convert images to grayscale (if needed).
- Extract LBP features for texture-based learning.
- Resize images to 128×128 pixels.
- Normalize and split the dataset into train/test sets.

---

### 4️⃣ Verify Data Loaders (Optional but Recommended)

Ensure the dataset is correctly loaded before training.

```bash
python test_pipeline.py
```

✅ This will:

- Load a small batch of images from the dataset.
- Pass data through the ProtoCapsNet pipeline.
- Check for shape mismatches or errors.

---

### 5️⃣ Train the Model

Train ProtoCapsNet using the dataset of choice:

For Birds Dataset

```bash
python main.py --train --epochs 50 --batch_size 32 --dataset birds --lr 0.001
```

For Cars Dataset

```bash
python main.py --train --epochs 50 --batch_size 32 --dataset cars --lr 0.001
```

✅ This will:

- Load the preprocessed dataset.
- Train the model using LBP + Capsule Network + K-Means Routing.
- Save the trained model for future evaluation.

---

### 6️⃣ Evaluate the Model

Check the model’s accuracy and loss on the test dataset.

For Birds Dataset

```bash
python main.py --evaluate --dataset birds
```

For Cars Dataset

```bash
python main.py --evaluate --dataset cars
```

✅ This will:

- Compute accuracy, loss, and class-wise performance.
- Ensure the model generalizes well to unseen images.

---

### 7️⃣ Make Predictions on New Images

Run inference on a new image using the trained model.

```bash
python main.py --predict --image_path "path/to/your/image.jpg"
```

✅ This will:

- Apply LBP preprocessing to the input image.
- Pass it through the Capsule Network.
- Predict the class label with confidence scores.

---

### 8️⃣ Visualize Model Predictions

Run the following command to visualize predictions:

For Birds Dataset

```bash
python visualize_predictions.py --dataset birds --num_images 8
```

For Cars Dataset

```bash
python visualize_predictions.py --dataset cars --num_images 8
```

✅ This will:

- Display test images with predicted vs actual labels.
- Show prototype-based explanations for model decisions.

---

## 📌 Project Workflow

###### 1️⃣ Install dependencies → `pip install -r requirements.txt`<br>

###### 2️⃣ Download & preprocess datasets → `python utils/data_preprocessing.py`<br>

###### 3️⃣ Train the model → `python main.py --train --dataset birds` <br>

###### 4️⃣ Evaluate the model → `python main.py --evaluate --dataset birds` <br>

###### 5️⃣ Make a prediction → `python main.py --predict --image_path "image.jpg" --dataset birds` <br>

###### 6️⃣ Visualize results → `python visualize_predictions.py --dataset birds` <br>

---

## 📌 Final Notes

- Always preprocess the dataset before training.
- Verify data loading using test_pipeline.py.
- Use correct dataset names (birds or cars) when running commands.
