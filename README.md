# QSAR-Modeling-and-Virtual-Screening-Pipeline
A full QSAR and virtual screening framework including data preparation, SMILES-to-SDF conversion, RDKit fingerprints, SHAP-based explainable models, and database screening. Designed for reproducible computational chemistry and drug discovery workflows.
It has been structured so that each step is modular, transparent, and easy to adapt to different datasets or project goals.
**#	Overview**
This project provides a full workflow for ligand-based drug discovery using Python and RDKit.
The idea is simple: start with a raw IC₅₀/SMILES dataset, clean it, generate fingerprints, build QSAR models, interpret them, and finally use the best model to screen external libraries.
All scripts follow consistent formatting, input/output conventions, and publication-quality plotting.
**#The repository includes:**
•	Data cleaning
•	Fingerprint generation
•	Fingerprint filtering
•	Batch QSAR modeling with automatic model selection
•	SHAP interpretability pipeline
•	Virtual screening of large chemical files
•	SMILES → SDF converter for downstream MD or docking workflows
**#	Pipeline Structure**
Below is a short explanation of each script so users understand exactly what it does and where it fits into the workflow.
**#2.1	Data_cleaning.py**
Clean ChEMBL-style bioactivity tables and prepare them for modeling. The script removes missing values, enforces nM units, computes pIC₅₀, applies an activity threshold, and removes duplicate SMILES entries. It also allows optional canonicalization to ensure consistent molecular identity. This step standardizes the dataset so later stages receive clean, reproducible inputs.
**#2.2	generate_fingerprints.py**
Generate RDKit molecular fingerprints from SDF files using both modern and fallback APIs.
It supports Morgan, MACCS, RDKit, AtomPair, and Torsion fingerprints, and attaches SD properties to the output table. Interactive menus make it easy to choose fingerprint types, and the output can be written in CSV or Parquet format.
**#2.3	filter_fingerprints.py**
Filter raw fingerprint matrices by removing constant bits, rare bits, highly correlated bits, and low-variance features.
This reduces noise, speeds up training, and improves generalization. The script preserves metadata columns so identifiers stay aligned through the workflow.
**#2.4	qsar_modeling_shap.py**
Train multiple QSAR models in parallel, evaluate performance, tune hyperparameters, and compute SHAP interpretability. The script supports scikit-learn, XGBoost, LightGBM, Keras, and PyTorch-skorch models. It automatically selects the best performing model, creates plots, and saves all outputs cleanly to an organized directory structure. This is the core of the pipeline and includes extensive logging, feature filtering, and memory-safe SHAP execution.
**#2.5	screen_database.py**
Use a trained QSAR model to screen external molecular databases (CSV/Parquet).
The script aligns fingerprint columns to avoid feature-shape mismatches and exports predictions, top-ranked molecules, and score histograms.
This step enables high-throughput virtual screening while ensuring reproducibility and feature consistency.
**#2.6	smiles_to_sdf.py**
Convert SMILES tables to SDF files with metadata preserved as SD tags. Supports optional conformer generation using ETKDG and MMFF/UFF optimization. Useful for downstream docking, MD, or 3D descriptor generation.
**#	Environment Files**
#Two complete environment files are provided:
•	QSAR_environment_cpu.yml - for CPU-only setups
•	QSAR_environment_gpu.yml - for accelerated training on NVIDIA GPUs
Both environments include RDKit, scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch, SHAP, and plotting dependencies.
**#To create the environment:**
conda env create -f QSAR_environment_cpu.yml
conda env create -f QSAR_environment_gpu.yml
**#Activate:**
conda activate QSAR_environment
**#	Workflow Summary**
1.	Clean your dataset
python Data_cleaning.py --in raw.xlsx --out clean.xlsx --add-pic50 --active-threshold 6.0
2.	Convert SMILES to SDF
python smiles_to_sdf.py --in clean.xlsx --out ligands.sdf
3.	Generate fingerprints
python generate_fingerprints.py --in ligands.sdf --out fingerprints.csv
4.	Filter fingerprints
python filter_fingerprints.py --in fingerprints.csv --out fingerprints_filtered.csv
5.	Run QSAR modeling with SHAP
python qsar_modeling_shap.py --config config.json
6.	Screen chemical databases
python screen_database.py --db zinc.csv --model models_out/
5	Citation and Copyright
This repository is copyright © 2025 Muhammad Waqas.
**#All rights reserved.**
The code is free to use only for academic and research purposes, but modification, redistribution, or commercial use is not permitted.
If you use any part of this pipeline, you must cite:
Waqas, M. (2025). QSAR Modeling and Virtual Screening Pipeline.
**#	Contact**
For questions, collaborations, or bug reports, please feel free to open an Issue or contact the author.

