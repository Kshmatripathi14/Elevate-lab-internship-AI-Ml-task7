# SVM Task 7 - Support Vector Machines (SVM)

This repository contains a complete example project for **Task 7: Support Vector Machines (SVM)**
from the uploaded assignment. It uses scikit-learn's built-in **Breast Cancer** dataset for binary
classification (malignant vs benign) so no external dataset download is required.

## What is included
- `data_loader.py` : loads and prepares the dataset (optionally scales & reduces to 2D for visualization)
- `train.py` : trains SVM with linear and RBF kernels, runs GridSearchCV for hyperparameter tuning, cross-validation, saves models & metrics
- `visualize.py` : visualizes decision boundaries (2D via PCA) and shows confusion matrix & ROC curve
- `requirements.txt` : Python package requirements
- `notebooks/` : Jupyter notebook demonstrating interactive usage (optional)
- `scripts/run_all.sh` : convenience script to run training and visualization
- `README.md` : this file
- `.gitignore` : common ignores
- `LICENSE` : MIT license

## How to run (local)
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Train & evaluate:
   ```bash
   python train.py --output_dir outputs
   ```
3. Visualize results (after training):
   ```bash
   python visualize.py --model_dir outputs
   ```

## How to push to GitHub
1. Create a new repo on GitHub (via website) named e.g. `svm_task7_project`.
2. From this project directory:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - SVM Task 7 project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/svm_task7_project.git
   git push -u origin main
   ```

## Notes
- The decision boundary visualization reduces the dataset to 2 principal components (PCA) for display.
- The training script performs GridSearchCV for both linear and RBF kernels and saves best models and metrics.
