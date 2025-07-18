# Gazimed: Early Alzheimer's Disease Detection System

**Advanced AI-powered early detection of Alzheimer's disease using paired MRI and PET brain imaging**

---

## 🧠 Clinical Overview

Gazimed is a state-of-the-art deep learning system designed to assist clinicians in the early detection of Alzheimer's disease. By analyzing paired T1-weighted MRI and ^18F-FDG PET brain scans alongside clinical features, our system provides continuous risk scores (0-1) to support clinical decision-making.

### Key Clinical Benefits

- **Early Detection**: Identifies Alzheimer's risk before severe cognitive decline
- **Multimodal Analysis**: Combines structural (MRI) and metabolic (PET) brain imaging
- **Clinical Integration**: Incorporates 118 clinical features for comprehensive assessment
- **Explainable AI**: Provides attention maps highlighting relevant brain regions
- **Validated Approach**: Built on established medical imaging datasets (ADNI, OASIS-3)

---

## 🔬 How It Works

```mermaid
flowchart TD
    A[MRI T1 Scan] --> D[Preprocessing Pipeline]
    B[PET FDG Scan] --> D
    C[Clinical Features<br/>118 parameters] --> E[Clinical Encoder]
    
    D --> F[3D Swin-UNETR<br/>Backbone]
    F --> G[Cross-Modal<br/>Attention Fusion]
    E --> G
    
    G --> H[Multimodal<br/>Feature Integration]
    H --> I[Risk Score<br/>0.0 - 1.0]
    
    I --> J[Clinical Report<br/>+ Attention Maps]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style I fill:#fff3e0
    style J fill:#fce4ec
```

### Technical Architecture

1. **Preprocessing**: N4 bias correction, MNI152 registration, normalization
2. **Feature Extraction**: 3D Swin Transformer with shifted window attention
3. **Multimodal Fusion**: Cross-attention between MRI/PET and clinical data
4. **Risk Assessment**: Continuous score output with confidence intervals
5. **Explainability**: Attention visualization for clinical interpretation

---

## 📊 Target Clinical Performance

We aim to achieve the following performance metrics through our validation studies:

| Metric | Target Performance | Clinical Significance |
|--------|-------------------|----------------------|
| **AUC-ROC** | 0.90+ | Excellent discrimination ability |
| **Sensitivity** | 85%+ | High detection of positive cases |
| **Specificity** | 85%+ | Low false positive rate |
| **Correlation** | r = 0.80+ | Strong agreement with clinical scores |

### Validation Datasets
- **ADNI**: Alzheimer's Disease Neuroimaging Initiative
- **OASIS-3**: Open Access Series of Imaging Studies
- **AIBL**: Australian Imaging, Biomarker & Lifestyle Study

---

## 🏥 Clinical Workflow Integration

```mermaid
sequenceDiagram
    participant C as Clinician
    participant G as Gazimed System
    participant P as PACS/Imaging
    participant R as Clinical Report
    
    C->>P: Order MRI + PET scans
    P->>G: Send DICOM images
    C->>G: Input clinical features
    G->>G: AI analysis
    G->>R: Generate risk assessment
    R->>C: Clinical report + visualizations
    C->>C: Clinical decision making
```

### Integration Points
- **PACS Integration**: Direct DICOM import from imaging systems
- **EMR Compatibility**: Clinical feature extraction from electronic records
- **Reporting**: Structured reports with attention visualizations
- **Quality Assurance**: Built-in validation and quality checks

---

## 🔍 Explainable AI Features

### Attention Visualization
The system provides clinically interpretable attention maps highlighting:

- **Hippocampus**: Memory formation and early AD pathology
- **Entorhinal Cortex**: Gateway to hippocampal formation
- **Posterior Cingulate**: Default mode network alterations
- **Precuneus**: Early metabolic changes in AD

### Clinical Validation Tools
- Anatomical region overlap metrics
- Attention consistency across similar cases
- Correlation with known AD biomarkers
- Comparison with radiologist assessments

---

## 🚀 Getting Started

### System Requirements

**Hardware:**
- GPU: NVIDIA RTX 3080 or better (12GB+ VRAM recommended)
- RAM: 32GB+ for processing full-resolution volumes
- Storage: 500GB+ for datasets and models

**Software:**
- Python 3.8+
- CUDA 11.8+
- Docker (optional, for containerized deployment)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/gazimed/alzheimers-detection.git
cd alzheimers-detection

# Install dependencies
pip install -e .

# Initialize database
gazimed-init --setup-database

# Download pretrained models
gazimed-download --model swin-unetr-pretrained
```

### Basic Usage

```python
from gazimed import AlzheimersDetector

# Initialize the detector
detector = AlzheimersDetector.from_pretrained('swin-unetr-v1.0')

# Analyze a case
result = detector.predict(
    mri_path='patient_001_T1.nii.gz',
    pet_path='patient_001_FDG.nii.gz',
    clinical_features=clinical_data
)

print(f"Alzheimer's Risk Score: {result.risk_score:.3f}")
print(f"Confidence Interval: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

# Generate clinical report
report = detector.generate_report(result)
report.save('patient_001_report.pdf')
```

---

## 📈 Model Performance Details

### Target Cross-Validation Results

```mermaid
graph LR
    A[5-Fold CV Target] --> B[Fold 1: AUC 0.90+]
    A --> C[Fold 2: AUC 0.90+]
    A --> D[Fold 3: AUC 0.90+]
    A --> E[Fold 4: AUC 0.90+]
    A --> F[Fold 5: AUC 0.90+]
    
    B --> G[Target Mean: 0.90+]
    C --> G
    D --> G
    E --> G
    F --> G
```

### Target Comparison with Baseline Methods

| Method | Target AUC | Target Sensitivity | Target Specificity | Notes |
|--------|------------|-------------------|-------------------|-------|
| **Gazimed (Goal)** | **0.90+** | **85%+** | **85%+** | Multimodal + Clinical |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Gazimed - Advancing early Alzheimer's detection through AI innovation* 🧠✨