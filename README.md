# Traffic Sign Recognition

## Getting Started

1. **Clone or download this repository.**
2. **Install dependencies:**
	 - Open a terminal in the project root and run:
		 ```pwsh
		 pip install -r requirements.txt
		 ```
3. **Prepare images:**
	 - Create a folder named `pics` in the project root.
	 - Add your images to the `pics` folder.

## Running the Pipeline

- To run the main pipeline:
	```pwsh
	python run_pipeline.py
	```
- To get raw segmentation model output:
	```pwsh
	python get_raw_output.py
	```

## Model Files
- Pretrained models should be placed in the `models` folder:
	- `classification_model.pth`
	- `segmentation_model.pth`

## Output
- Results will be saved in the `results` folder.

## Project Structure
```
classification_model.py
get_raw_output.py
run_pipeline.py
segmentation_model.py
models/
	classification_model.pth
	segmentation_model.pth
pics/
results/
requirements.txt
README.md
```

## Notes
- Ensure you have Python 3.8+ installed.
- If you encounter missing package errors, install them using `pip install <package>`.