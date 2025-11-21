# PulmoVision

PulmoVision is an end-to-end 3D Slicer module designed for automated radiomic analysis of CT lung tumors. The system integrates preprocessing, AI-based segmentation, post-processing, radiomics feature extraction, and visualization into a single reproducible pipeline. This project implements a lightweight 3D U-Net trained on synthetic CT lung tumor volumes and embeds it directly into a Slicer workflow. The goal is to demonstrate how standardized, integrated pipelines can improve reproducibility and usability in radiomics research.

## Features

- End-to-end radiomics pipeline for CT lung tumor analysis
- Lightweight 3D U-Net segmentation integrated directly into the workflow
- Standardized preprocessing including resampling, windowing, and normalization
- Automated post-processing and mask cleanup
- Radiomics feature extraction, including geometric and intensity-based statistics
- One-click execution via a custom scripted 3D Slicer module
- Modular design allowing replacement with pretrained or clinically validated models

## Repository Structure

    pulmovision/
    │
    ├── pulmovision/          # Core Python package (training, inference, preprocessing)
    ├── slicer_module/        # 3D Slicer scripted module implementation
    ├── data/                 # Sample CT and synthetic datasets
    ├── experiments/          # Training logs and model checkpoints
    └── docs/                 # Proposal, diagrams, internal notes

## Installation

### Python environment

    conda env create -f environment.yml
    conda activate pulmovision

or

    pip install -r requirements.txt

### 3D Slicer Module Installation

1. Clone or download this repository.
2. Open 3D Slicer.
3. Navigate to Edit → Application Settings → Modules.
4. Add the directory "slicer_module/PulmoVision" to the module search path.
5. Restart Slicer.
6. The PulmoVision module will appear in the module list.

## Enabling UNet3D Segmentation (PyTorch Support)
The **UNet3D** and **Auto** segmentation modes require **PyTorch**.  
3D Slicer does *not* include PyTorch by default, so it must be installed inside Slicer.

1. Open **Extensions Manager** in Slicer  
2. Search for **“PyTorch”**  
3. Install the **PyTorch** extension  
4. Restart Slicer  
5. Verify installation inside the Python Interactor:

```python
import PyTorchUtils
torch = PyTorchUtils.PyTorchUtilsLogic().torch
print("PyTorch version:", torch.__version__)
```

If this runs without errors, PyTorch is active and UNet3D segmentation is enabled.

### Running the Full Pipeline (3D Slicer)

1. Load a CT volume into Slicer.
2. Open the PulmoVision module.
3. Select the input volume.
4. Click "Run Full Pipeline".
5. View tumor segmentation overlays and extracted radiomic features.


### Running Standalone Inference (Python)

    from pulmovision.pipeline import run_full_pipeline
    mask, features = run_full_pipeline("path/to/ct.nrrd")

## License

This project is released under the MIT License.

## Authors

James Mascarenhas  
Jihyeon Park  
CISC 881: Medical Informatics  
Queen's University (2025)

