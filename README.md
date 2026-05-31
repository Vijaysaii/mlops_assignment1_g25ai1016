# MLOps Assignment 1 - House Price Prediction

This project trains machine learning models to predict house prices using the Boston Housing dataset.

## Setup Instructions

### 1. Create and activate conda environment

```bash
conda create -n mlops_a1 python=3.10
conda activate mlops_a1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the models

**Decision Tree Regressor:**
```bash
python train.py
```

**KernelRidge Regressor:**
```bash
python train2.py
```

## Project Structure

- `misc.py` - Shared helper functions (load, preprocess, train, evaluate)
- `train.py` - Decision Tree model training script
- `train2.py` - KernelRidge model training script
- `requirements.txt` - Python dependencies

## Branches

- `main` - Contains README
- `dtree` - Decision Tree implementation
- `kernelridge` - KernelRidge implementation with GitHub Actions CI
