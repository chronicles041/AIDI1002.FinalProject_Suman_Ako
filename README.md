# DeiT: Data-efficient Image Transformers

## Overview

This repository provides code and instructions for implementing and evaluating the DeiT (Data-efficient Image Transformers) model introduced by Hugo Touvron et al. in 2020. DeiT is a vision transformer model that achieves high performance on image classification tasks with reduced data and computational requirements compared to traditional Convolutional Neural Networks (CNNs). This repository is designed to help users reproduce the results from the paper, test the model on new datasets, and experiment with hyperparameters and transfer learning. 
 ```bash
   https://github.com/facebookresearch/deit/blob/main/README_deit.md
   ```

## Aim

The primary aim of this project is to:
- Reproduce the results of the DeiT model as presented in the original paper.
- Evaluate the performance of DeiT on different datasets beyond ImageNet.
- Experiment with hyperparameter tuning and transfer learning to assess model flexibility and applicability.

## Requirements

To ensure compatibility, the following versions of packages are required:
- `torch==1.13.1`
- `torchvision==0.14.1`
- `timm==0.6.12`

### Setting Up the Environment

1. **Clone the Repository**

   Begin by cloning the repository to your local machine:

   ```bash
   git clone https://github.com/chronicles041/AIDI1002.FinalProject_Suman_Ako
   ```

2. **Navigate to the Project Directory**

   Change into the project directory:

   ```bash
   cd your-repository
   ```

3. **Create and Activate a Virtual Environment (Recommended)**

   Create a virtual environment to manage dependencies and avoid conflicts:

   ```bash
   python -m venv myenv
   ```

   Activate the virtual environment:
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```

4. **Install Required Packages**

   Install the specific versions of the required packages:

   ```bash
   pip install torch==1.13.1 torchvision==0.14.1 timm==0.6.12
   ```

5. **Download Additional Files**

   Download the ImageNet category names file for nicer display of results:

   ```bash
   wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
   ```

6. **Install Jupyter Notebook (If Not Installed)**

   If you don’t have Jupyter Notebook installed, you can add it using:

   ```bash
   pip install notebook
   ```

7. **Run Jupyter Notebook**

   Start Jupyter Notebook to access and run the provided notebook:

   ```bash
   jupyter notebook
   ```

   Open the Jupyter notebook file (`your_notebook.ipynb`) in the browser and follow the instructions within the notebook.

## Project Implementation

1. **Reproduce Results**
   - Follow the steps outlined in the notebook to set up your environment and reproduce the results from the DeiT paper using the ImageNet dataset.

2. **Evaluate on New Datasets**
   - Test the DeiT model on additional datasets such as CIFAR-10 or CIFAR-100.
   - Compare the performance metrics with those reported in the paper.

3. **Hyperparameter Tuning**
   - Experiment with different hyperparameters, including learning rate, batch size, and number of epochs, to optimize model performance.

4. **Transfer Learning**
   - Fine-tune the DeiT model on specific domain datasets (e.g., medical images) to assess its transfer learning capabilities.

## Contribution

TO BE DONE


## Contact
## BibTeX

```bibtex
@InProceedings{pmlr-v139-touvron21a,
  title =     {Training data-efficient image transformers & distillation through attention},
  author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
  booktitle = {International Conference on Machine Learning},
  pages =     {10347--10357},
  year =      {2021},
  volume =    {139},
  month =     {July}
}



