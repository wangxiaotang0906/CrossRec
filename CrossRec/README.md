# scCrossRec: Code Repository

Welcome to the official code repository of **scCrossRec**.
This repository provides complete, reproducible code for data preprocessing, graph construction, model training, and regulatory interpretation.

Because scCrossRec integrates **single-cell multi-omics**, **graph representation learning**, and **recommendation systems**, you may confuse at first time.
Don’t worry — we provide detailed, step-by-step instructions to help you get started quickly.

## Directory Structure
```
CrossRec/
│
└── original_datasets/                          # Public raw datasets in processed AnnData format
│   ├── ISSAAC-seq/
│   ├── 10xMultiome-PBMC/
│   ├── SHARE-seq/
│   ├── SNARE-seq/
│   └── NeurIPS2021_Multiome/
│ 
├── interaction_graph_generator/
│   ├── missing_attribute_generator/
│   │   ├── get_chorm.py                        # Optional if chorm information about ATAC or RNA is missing
│   │   ├── get_DNA_sequence.py                 # Optional if DNA_sequence information about ATAC is missing
│   │   ├── hg38.fa                             # Supporting files for DNA_sequence
│   │   └── gencode.v43.annotation.gtf          # Supporting files for chorm information
│   ├── interaction_graph_generator_ISSAAC.py
│   ├── interaction_graph_generator_PBMC.py
│   ├── interaction_graph_generator_SHARE.py
│   ├── interaction_graph_generator_SNARE.py
│   └── interaction_graph_generator_Multiome.py
│
└── interaction_graph_datasets/                 # Output directory for generated graph datasets
│   ├── ISSAAC/
│   ├── PBMC/
│   └── ....
│ 
├── environment.yml                             # environment file
├── model.py                                    # scCrossRec model
├── train.py                                    # Training script
├── regulatory_discovery_inference.py           # Inference and attention extraction for biological interpretability
├── regulatory_discovery_cell.ipynb             # Cell-level regulatory visualization
└── regulatory_discovery_cell_type.ipynb        # Cell-type–level regulatory maps, and compared to TF-binding-based regulatory maps
```

## Requirements
scCrossRec is implemented using:
- **Python 3.10.15**
- **PyTorch 2.5.1**
- **PyTorch Geometric (PyG) 2.6.1** for graph learning components

You can directly clone our environment using the provided environment file (recommended):
> CrossRec/environment.yml

## Pipeline Overview
### Step1: Preprocessing oringal datasets.
All datasets are provided in **AnnData** format, where:
- `rna_anndata.var` must include:
  `{"means", "variances_norm", "chrom", "strand", "highly_variable"}`
- `atac_anndata.var` must include: 
  `{"chrom", "dna_sequence"}`

You can download our preprocessed AnnData (ATAC_AnnData.h5ad, RNA_AnnData.h5ad, and DNA_sequences.fa) from the link below and unzip the "original_datasets.rar" file under "CrossRec/":
> https://drive.google.com/file/d/1QkBkcKbZ7ttx8crT9jJMZ7n1M-pyiGrR/view?usp=drive_link

If you use custom datasets, follow our format and use scripts under "CrossRec/interaction_graph_generator/missing_attribute_generator/" to fill missing attributes. You may need to download the supporting files "hg38.fa" and "gencode.v43.annotation.gtf" from:
> https://drive.google.com/file/d/1fOG_4zBMV9td7Rjf8IGKQOzqzT6x_IfG/view?usp=drive_link
> https://drive.google.com/file/d/1QBI1QqQ-Yz7c1C94UUvyZjfdMMIckedK/view?usp=drive_link

### Step2: Generate interaction graphs (graph datasets)
Use the corresponding script "CrossRec/interaction_graph_generator/interaction_graph_generator_{dataset}.py" to generate the interaction graph datasets.
Make sure to modify the `dataset_store_path` inside each script so that it matches your local directory structure.
Several optional parameters are important:
-   `select_cell_type = False     # Set to True if you want to restrict the dataset to specific cell types.`
-   `selected_cell_type = [".."]`
-   `for_discovery = False # If for_discovery is True, only positive samples are generated. This is commonly used when preparing a single cell type for regulatory discovery using a pretrained model`

### Step3: Training models
Train scCrossRec using "train.py".
Ensure that the dataset loading paths are correct for your directory layout, and adjust the model hyperparameters according to your experimental needs.

### Step4: Biological interpretability
After training, run "regulatory_discovery_inference.py" on your selected dataset.
This script generates cell-level attention matrices and additional regulatory information, which will be saved to the specified output directory.

To visualize the results:
- Use "regulatory_discovery_cell.ipynb" for cell-level regulatory matrices
- Use "regulatory_discovery_cell_type.ipynb" for cell-type regulatory maps

