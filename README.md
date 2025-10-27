# Chest X-ray Pneumonia Classifier

A production-quality deep learning project to detect Pneumonia from chest X-ray images, with a ready-to-use Streamlit web app and reproducible Jupyter notebooks for training, evaluation, and explainability.

## Overview
This project builds a binary image classifier that labels chest radiographs as NORMAL or PNEUMONIA. It leverages transfer learning on a curated chest X-ray dataset, includes Grad-CAM visualizations for interpretability, and provides a streamlined user interface for rapid triage and educational demonstrations.

## Key Features
- Streamlit web app for quick, interactive inference
- Transfer learning–based CNN with fine-tuning for strong generalization
- Grad-CAM heatmaps to visualize model attention
- Reproducible notebooks for training, evaluation, and ablations
- Clean project structure and pinned dependencies

## Performance Metrics
- Overall accuracy: ~95% on held-out test set
- Precision/Recall/F1 per class (reported in notebooks)
- Confusion matrix, ROC-AUC, PR curves and loss/accuracy curves provided
- Note: Metrics vary by split, seed, augmentation, and preprocessing

## Demo
- Live demo: https://chest-xray-pneumonia-classifier-arpan.streamlit.app/
- Screenshot/GIFs optional: add to a docs/ folder and reference here

## Technical Stack
- Language: Python 3.9–3.11
- Core DL: TensorFlow/Keras or PyTorch (per notebook implementation)
- Serving/UI: Streamlit
- Metrics/Model utils: scikit-learn, NumPy, Pandas
- Visualization: Matplotlib, Seaborn (optionally Plotly)
- Imaging: Pillow, OpenCV

## Dataset
- Source: Chest X-Ray Images (Pneumonia) dataset (Kermany et al., 2018; Kaggle)
- Classes: NORMAL, PNEUMONIA
- Modalities: AP/PA chest radiographs (grayscale)
- Licensing: Refer to the dataset’s original license and usage terms

Data is not included in this repository. After download, organize locally as:

```
data/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

## Project Structure
```
.
├── app.py                        # Streamlit app
├── diagnosis-of-pneumonia.ipynb  # Training/evaluation notebook (v1)
├── diagnosis-of-pneumonia-v2.ipynb # Training/evaluation notebook (v2)
├── requirements.txt              # Dependency pins
├── README.md                     # Project documentation
└── data/                         # Not tracked; see Dataset section
```

## Installation
1) Clone the repo
```
git clone https://github.com/arpanpramanik2003/chest-xray-pneumonia-classifier.git
cd chest-xray-pneumonia-classifier
```

2) Create a virtual environment
```
# Conda (recommended)
conda create -n cxr-pneumo python=3.10 -y
conda activate cxr-pneumo

# Or venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3) Install dependencies
```
pip install -r requirements.txt
```

## Usage
### A) Run the Streamlit app locally
```
streamlit run app.py
```
Then open the local URL shown in the terminal:
- Upload a chest X-ray image (JPG/PNG)
- View predicted label with probability
- Optionally view Grad-CAM visualization (if enabled)

### B) Train/evaluate with notebooks
Open and run:
- diagnosis-of-pneumonia.ipynb
- diagnosis-of-pneumonia-v2.ipynb

Typical steps include:
- Data loading, resizing, normalization, augmentation
- Model definition (transfer learning), optimizer/loss setup
- Training with early stopping/checkpointing
- Evaluation: metrics, confusion matrix, ROC/PR
- Explainability: Grad-CAM

Update dataset paths in the first cells to match your local data directory.

## Requirements
See requirements.txt for exact versions. Core packages include:
- streamlit
- tensorflow/keras or torch/torchvision
- scikit-learn
- numpy, pandas
- pillow, opencv-python
- matplotlib, seaborn

## Contributing
Contributions are welcome!
- File an issue to discuss proposals or bugs
- Use feature branches and descriptive commit messages
- Include benchmarks, screenshots, and reproduction steps when relevant

## Important Notes
- Not a medical device; for research and education only
- Do not use as the sole basis for clinical decisions
- Performance may vary across institutions, devices, and populations
- Ensure patient privacy and compliance with relevant regulations

## License
This project is released under the MIT License. See the repository’s MIT license section for details.

## Contact
- Author: Arpan Pramanik
- Demo: https://chest-xray-pneumonia-classifier-arpan.streamlit.app/
- Issues: https://github.com/arpanpramanik2003/chest-xray-pneumonia-classifier/issues
