> A deep learning project for classifying drones using transfer learning.  
> Designed to be simple, fast, and beginner-friendly.


# Drone Image Classification using MobileNetV2

This project uses **transfer learning** to classify images into four categories:
- `dji_inspire`
- `dji_mavic`
- `dji_phantom`
- `no_drone`

Instead of training a model from scratch, we use a pre-trained model called **MobileNetV2** that already knows how to recognize general shapes and patterns. We then teach it to recognize different types of drones (or no drone at all) using our custom dataset.

We use **TensorFlow/Keras** to load images, prepare the data, build the model, train it, and evaluate how well it performs.

---

## Dataset Structure

Make sure your dataset is organized like this:

Synthetic_Drone_Classification_Dataset/ 
├── train/ │ 
            ├── dji_inspire/ │ 
            ├── dji_mavic/ │ 
            ├── dji_phantom/ │ 
            └── no_drone/ 
└── val/ 
            ├── dji_inspire/ 
            ├── dji_mavic/ 
            ├── dji_phantom/ 
            └── no_drone/



---


Each subfolder contains images of that specific class. The folder names are used automatically as labels for training.

---

##  How It Works (Simple Overview)

This project is based on the idea that we don’t need to start from zero to train a model.  
We take a model that already learned how to "see" (MobileNetV2, trained on millions of images), and we just **retrain its final layer** to detect drones instead of dogs, cats, or cars.

This method is called **transfer learning**, and it’s fast and very effective — especially when we don’t have a huge dataset.

---

##  Project Workflow

### 1. Load Image Paths

We define the folder paths where our training and validation images are stored. This tells the model where to find the data.

---

### 2. Image Preprocessing and Augmentation

We use `ImageDataGenerator` to:
- Scale image pixel values from `[0–255]` to `[0–1]` (normalization)
- Apply random changes like rotation, zoom, and flipping to the training images  
  -> This helps the model not memorize but **learn to generalize**
- Validation images are only normalized, with no changes

---

### 3. Create Data Generators

We use a function that:
- Automatically assigns labels based on folder names
- Loads images in **batches**
- Resizes them to 128×128 pixels to match the model's input shape

This makes training faster and memory-efficient.

---

### 4. Use a Pretrained Model (Transfer Learning)

We load **MobileNetV2**, a model trained on ImageNet (a huge dataset).  
We remove its last layer (which classifies 1000 objects) and **freeze the rest of the model**, so it doesn’t “forget” what it has already learned.

---

### 5. Build a Custom Classification Head

We add new layers on top:
- A pooling layer to reduce dimensions
- A dropout layer (randomly turns off neurons during training to avoid overfitting)
- A final layer with 4 outputs (1 for each class), using **softmax activation** so it returns probabilities

---

### 6. Compile the Model

We prepare the model for training by setting:
- **Adam optimizer** – a smart and popular optimizer
- **Categorical crossentropy** – the loss function used for multi-class classification
- **Accuracy** – to measure how often the model is right

---

### 7. Train the Model

We train the model using `.fit()`:
- The model sees the training images in batches and learns from them
- After each round (called an **epoch**), it checks how well it performs on validation images

This helps us monitor progress and catch problems like overfitting.

---

### 8. Visualize Accuracy

We plot how **accurate** the model is:
- `Training accuracy`: how well the model does on images it has seen
- `Validation accuracy`: how well it does on new, unseen images

If the training accuracy is much higher than validation accuracy, that means **overfitting** (the model is memorizing instead of learning).

---

### 9. Visualize Loss

We also plot the **loss**, which measures:
- How far the model’s predictions are from the correct answers
- Lower loss means better performance

If training loss is low but validation loss is high - that's a sign of overfitting.

---

##  Understanding Accuracy vs. Loss

| Metric   | What It Tells You                                 |
|----------|---------------------------------------------------|
| Accuracy | How often the model predicts the correct class    |
| Loss     | How far off the model’s predictions are, even if correct |

**Even with high accuracy, a high loss means the model might be uncertain or overly confident in wrong guesses.**

---

## Custom Image Prediction

Once the model is trained, I tested it on some random images from the internet:
- The images were resized and normalized to 128×128
- The model predicted class probabilities
- If confidence was high, it returned the most likely class
- Otherwise, it returned: “Uncertain”

I even tried it with a picture of a cat, and it responded:
*"This image does not clearly belong to any known class."*

So I think the model performs pretty well ;). 

