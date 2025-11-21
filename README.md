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

## Usage

### Running the Full Pipeline (3D Slicer)

1. Load a CT volume into Slicer.
2. Open the PulmoVision module.
3. Select the input volume.
4. Click "Run Full Pipeline".
5. View tumor segmentation overlays and extracted radiomic features.

#### Enabling the UNet3D / Auto segmentation methods

The UNet3D-backed methods require PyTorch, which is not bundled with the stock Slicer Python.
Install a CPU-only PyTorch build directly inside Slicer using the Python Interactor (then restart Slicer):

```
slicer.util.pip_install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
```

After restarting, the **auto** and **unet3d** options will load a lightweight checkpoint from
`slicer_module/PulmoVision/PulmoBackend/checkpoints/unet3d_synthetic.pt`, or you can point the
`Weights path` parameter at your own trained model.

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

