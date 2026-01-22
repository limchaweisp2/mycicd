# My CICD Project

This project demonstrates a fully automated machine learning training pipeline using DVC for pipeline management and CML for reporting metrics and plots to GitHub Pull Requests.

## Repository Structure

- `train.py`: Python script to generate synthetic data and train a RandomForest model.
- `dvc.yaml`: DVC pipeline configuration.
- `.github/workflows/cml.yaml`: GitHub Actions workflow for CI/CD and CML reporting.
- `requirements.txt`: Python dependencies.

## Local Setup and Initialization

To initialize this repository locally, follow these steps:

1. **Initialize Git repository:**
   ```bash
   git init
   ```

2. **Initialize DVC:**
   ```bash
   dvc init
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add files to Git:**
   ```bash
   git add .
   git commit -m "Initial commit: Set up DVC pipeline and CML workflow"
   ```

5. **Run the pipeline locally (optional):**
   ```bash
   dvc repro
   ```

## Triggering the CI/CD Pipeline

1. **Create a new branch:**
   ```bash
   git checkout -b feature/experiment
   ```

2. **Make changes to `train.py` or other files.**

3. **Commit and push to GitHub:**
   ```bash
   git add .
   git commit -m "Update training logic"
   git push origin feature/experiment
   ```

4. **Open a Pull Request** on GitHub. The GitHub Action will trigger, run the DVC pipeline, and CML will post a comment with the metrics and the confusion matrix plot directly on your PR.
