# **Image2Biomass – Deep Learning Project**

This repository contains our team’s work for the Image2Biomass Kaggle competition, where the goal is to predict five biomass targets from RGB images of grass.
Due to the very small dataset (357 training images), this project focuses heavily on data augmentation, transfer learning, and careful model experimentation.

We approached this project as a friendly internal challenge:
Each of us built our own model from scratch, and whoever achieved the best Kaggle score would have their model chosen as the “official” one for our presentation and final video.

**Winner:** **Adrian Ramirez**, achieving a score of ~0.30
His model is documented in the Official model folder.

---

# **Final Presentation Video**

**YouTube Video:** [https://youtu.be/ggFitAc4BQs](https://youtu.be/ggFitAc4BQs)

The video explains:

* the Image2Biomass problem
* our model architecture
* training and deployment workflow
* experiments and results
* and our reflections on improving the model

---

# **Repository Structure**

```
IMAGE2BIOMASS/
│
├── Official model/              # The winning model (Adrian's)
│   ├── datasets2.py             # Data loading + augmentation code
│   ├── main3.py                 # Script to train/evaluate model
│   ├── models3.py               # Model definitions (ResNet18 + MLP head)
│   ├── image2biomass-presstek.ipynb   # Final official notebook used for the submission
│   ├── version_101/             # Kaggle-compatible version folder
│   ├── submission_presstek.csv  # Kaggle submission file
│   └── requirements.txt         # Python dependencies
│
├── OTHER SUBMISSIONS/           # Individual attempts by team members
│   ├── crptk/                   # Edrees' submission models
│   └── eshan/                   # Eshan's submission models
│
├── README.md
└── requirements.txt             # Root-level dependency file
```

---

# **Project Overview**

The task:
Predict **5 biomass values** from RGB images of vegetation using deep learning.

Challenges:

* Very small dataset
* High variance in lighting and angle
* Multi-output regression
* Need for strong generalization on unseen data

Solution:

* Resize images to 224×224
* Apply aggressive data augmentation (flips, rotations, jitter)
* Use **ResNet18** pretrained on ImageNet as a feature extractor
* Build a custom **MLP regression head**
* Train and evaluate multiple model variations internally
* Select the best-performing solution for final presentation

---

# **Official Model (Winning Approach)**

The winning model (“Presstek”) uses:

### **Backbone**

* **ResNet18 pretrained** on ImageNet
* First layers extract edges/textures
* Residual blocks expand features from 64 → 128 → 256 → 512
* Global average pooling to produce a 512-dimensional feature vector

### **Custom Classifier Head**

* Flatten → [Batch, 512]
* Linear (512 → 64) + ReLU
* Dropout
* Linear (64 → 5) for multi-output regression

### **Training Setup**

* MSE loss
* Adam / AdamW optimizer
* Strong augmentation to combat overfitting
* Careful learning rate selection
* Validation splits for tuning

### **Deployment**

* Load trained model
* Apply same preprocessing to Kaggle test set
* Run inference → produce submission.csv

---

# **Other Experiments**

The `OTHER SUBMISSIONS/` folder contains individual experimental models by:

* **Edrees (crptk)**
* **Eshan**

These were part of a competition-style workflow to see who could produce the best Kaggle score.
The best-performing model became the “official” version.

---

# **How to Run the Official Model**

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train or evaluate using scripts

```
python main3.py
```

### 3. Or open the Jupyter Notebook

`image2biomass-presstek.ipynb` contains the full pipeline including:

* preprocessing
* training
* validation
* generating submission files

---

# **Future Improvements**

Some improvements we identified:

* Use larger backbones (ResNet34, ResNet50, EfficientNet)
* Fine-tune the backbone instead of freezing it
* Use regression-friendly loss functions (Huber, L1+L2 combos)
* Try Mixup/CutMix for synthetic image generation
* Hyperparameter tuning (learning rate, batch size)
* Collect more real data

---

# **Team Members**

* **Adrian Ramirez** (official model)
* **Edrees Amiri**
* **Eshan**
