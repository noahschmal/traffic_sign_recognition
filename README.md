# Traffic Sign Recognition

This project implements a two-stage pipeline for traffic sign recognition. First, a segmentation model identifies the locations of traffic signs in an image. Then, a classification model identifies the type of each detected sign.

## Getting Started

1.  **Clone or download this repository.**
2.  **Install dependencies:**
    -   Open a terminal in the project root and run:
        ```bash
        pip install -r requirements.txt
        ```
3.  **Prepare images:**
    -   Create a folder named `pics` in the project root.
    -   Add your images to the `pics` folder.

## Running the Pipeline

-   To run the main pipeline (segmentation followed by classification):
    ```bash
    python run_pipeline.py
    ```
-   To get raw segmentation model output:
    ```bash
    python get_raw_output.py
    ```

## Model Files

-   Pretrained models are located in the `models` folder:
    -   `segmentation_model.pth`: The default U-Net segmentation model.
    -   `classification_model.pth`: The ResNet-50 based classification model.
    -   `model_efficientnet.pth`: An EfficientNet-U-Net segmentation model (alternative).
    -   `model_filtered.pth`: A U-Net segmentation model trained on a filtered dataset (alternative).

## Training

The following scripts are available for training the models:

-   `train_filtered.py`: Trains the U-Net segmentation model (`model.py`) using a filtered dataset.
-   `train_efficientnet.py`: Trains the EfficientNet-U-Net segmentation model (`model_efficientnet.py`).

To run a training script:
```bash
python <script_name>.py
```

## Model Architecture Files

-   `model.py`: Defines a standard U-Net architecture for segmentation. This is identical to `segmentation_model.py`.
-   `segmentation_model.py`: Defines a standard U-Net architecture for segmentation. This is identical to `model.py`.
-   `classification_model.py`: Defines a ResNet-50 based architecture for classification.
-   `model_efficientnet.py`: Defines a U-Net-like architecture with an EfficientNet-B0 encoder for segmentation.

## Jupyter Notebook

-   `TrafficSignClassifier.ipynb`: A Jupyter Notebook that provides a complete workflow for the classification task, including:
    -   Dataset preparation and preprocessing.
    -   Training the ResNet-50 classification model.
    -   Evaluating the model with metrics like accuracy, F1-score, and a confusion matrix.
    -   Error analysis and visualization of misclassified examples.
    -   A function to test the model on custom images.

## Project Structure

```
.
├── models/
│   ├── classification_model.pth
│   └── segmentation_model.pth
├── pics/
├── results/
├── .gitignore
├── classification_model.py
├── get_raw_output.py
├── model_efficientnet.py
├── model.py
├── README.md
├── requirements.txt
├── run_pipeline.py
├── segmentation_model.py
├── TrafficSignClassifier.ipynb
├── train_efficientnet.py
└── train_filtered.py
```

## Notes

-   Ensure you have Python 3.8+ installed.
-   If you encounter missing package errors, install them using `pip install <package>`.