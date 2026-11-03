# AI-Driven Multi-Task CNN for Concurrent Hepatic Fibrosis Staging and Focal Lesion Characterization on Grayscale Ultrasound

This repository contains the official production-grade implementation of a lightweight Multi-Task Deep Learning framework designed for automated hepatic tissue characterization, multi-stage liver fibrosis scoring (F0–F4), and binary focal liver lesion differentiation (Hepatocellular Carcinoma vs. Hepatic Hemangioma). 

Optimised specifically for low-resource point-of-care environments and handheld B-mode grayscale ultrasound edge probes, this solution bridges the gap between raw medical hardware telemetry and standard diagnostic digital health frameworks.

## 🌟 Key Architectural Features
- **Lightweight CNN Architecture:** Tailored with optimized convolutional layers (scaling features from 32 to 192 dimensions across 8 residual blocks) deploying exactly 2,149,833 parameters for ultra-low latency inference on resource-constrained handheld systems.
- **Asymmetric Loss Masking:** Features structured data manifests using a target index matrix mask of `-100` to completely bypass gradient derivations on unannotated tasks across hybrid clinical datasets.
- **Data Contamination Control:** Deploys a robust 64-bit difference hashing (`dHash` with a strict Hamming distance threshold ≤ 2) combined with SHA-256 cryptographic checking to isolate near-duplicate structures and prevent cross-split dataset leakage.

## 🛠️ Tech Stack & Requirements
- **Language Profile:** Python 3.10+
- **Deep Learning Framework:** TensorFlow / Keras (or PyTorch equivalent architecture layer mappings)
- **Pre-Processing & Topology Engines:** OpenCV, SciPy (Topological Peak Finders & Noise Filters), NumPy, Scikit-Image
- **Data Analysis & Visualization Logs:** Pandas, SQLite Database (Configured with automated B-Tree indexing benchmarks), Matplotlib, Seaborn

---

## 📊 Empirical Telemetry & Diagnostic Visualizations

### 1. Dual-Task Shared Optimization Tracking
The system convergence loops were controlled using an early-stopping patience boundary of 12 epochs driven strictly by the task-mean validation macro-F1 targets. Standalone baselines stabilized at Epoch 45 (fibrosis) and Epoch 34 (lesion), while multi-task joint optimization peaked smoothly at Epoch 56 out of a maximum 60-epoch budget.

<p align="center">
  <img src="https://github.com" width="75%" alt="Model Training Loss and Mean Validation Macro-F1 History History Mappings"/>
</p>

### 2. Primary Diagnostic Evaluation Matrices
Evaluating empirical classification performance constraints on frozen image-level held-out test splits. Shared feature extraction achieved an overall diagnostic accuracy of 71.02% for five-stage parenchymal tissue staging and a 97.51% boundary for focal lesion characterization.

<p align="center">
  <img src="https://github.com" width="60%" alt="Concurrent Model Hepatic Fibrosis Confusion Matrix Metrics Grid"/>
</p>

### 3. Quantitative Performance Distributions
Validation tracking curves display high stability. The single-task baseline lesion configuration yields an image-level accuracy threshold of 98.50% (395/401), while joint multi-head representational learning stabilizes focal lesion grading tasks at a balanced macro-F1 profile of 0.9750.

<p align="center">
  <img src="https://github.com" width="75%" alt="Focal Tumor Staging Metrics Precision Recall and F1 Distributions Data Plots"/>
</p>

### 4. Qualitative Interpretability & Grad-CAM Heatmaps
Grad-CAM evaluations are hosted strictly for qualitative model attention tracking and saliency inspection inside B-mode ultrasound frames. Saliency mappings verify feature extraction patterns across target lesion zones and progressive parenchymal tissue alterations.

<p align="center">
  <img src="https://github.com" width="48%" style="margin-right: 2%;" alt="Grad-CAM Hidden Layer Attention Tracking for Benign Hepatic Hemangioma Pathology"/>
  <img src="https://github.com" width="48%" alt="Grad-CAM Saliency Output Mapping for Progressive Hepatic Parenchymal Alterations Staging Baseline"/>
</p>

*Cautionary Notice: Grad-CAM vectors denote visual region-of-interest attention weights within the convolutional hidden layers and do not constitute formal absolute spatial localization bounds or independent diagnostic clinical confirmation.*

---

## 🔒 Clinical Constraints & Medical Data Privacy Compliance
*To maintain compliance with national patient safety ethics, institutional data confidentiality frameworks, and global hospital tracking regulations, the raw diagnostic B-mode ultrasound imagery arrays used during training loops are strictly confidential and are not hosted within this public code portal environment.*

## 👤 Author & Professional Background
**Independent Healthcare AI Architect & Biomedical Engineer**
- **10+ Years of Hands-on Clinical Experience** in core Biomedical Engineering pipelines and critical Hospital Systems Management infrastructure.
- **Certified Expertise Profiles:** Data Analytics Essentials & Python (Cisco Integration Frameworks), Fundamentals of Healthcare Systems Management.
- **Quality Engineering Standards:** Specializing in **ISO 13485** Medical Devices Quality Management Systems (QMS) protocols for global Software as a Medical Device (SaMD) structural integrations.
- **Research Profile:** Focused on developing software as a medical device (SaMD) with previous publications in neurological image analysis (Alzheimer's pathology detection) and assistive UI frameworks (Eye Mouse systems).
