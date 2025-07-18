"""
Setup configuration for Gazimed - Early Alzheimer's Disease Detection System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    requirements = [
        # Core ML/DL frameworks
        "pytorch-lightning>=2.0.0",
        "monai>=1.2.0",
        
        # Medical imaging
        "nibabel>=5.0.0",
        "SimpleITK>=2.2.0",
        "dicom2nifti>=2.4.0",
        
        # Data processing
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        
        # Database
        "sqlalchemy>=2.0.0",
        
        # Experiment tracking
        "mlflow>=2.5.0",
        "optuna>=3.2.0",
        "tensorboard>=2.13.0",
        
        # API and deployment
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        
        # Utilities
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        
        # Testing
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ]
    return requirements

setup(
    name="gazimed",
    version="0.1.0",
    author="Gazimed Research Team",
    author_email="research@gazimed.ai",
    description="Early Alzheimer's Disease Detection using Deep Learning on MRI/PET Brain Imaging",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/gazimed/alzheimers-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "gazimed-train=gazimed.training.cli:train_command",
            "gazimed-evaluate=gazimed.evaluation.cli:evaluate_command",
            "gazimed-predict=gazimed.deployment.cli:predict_command",
            "gazimed-serve=gazimed.deployment.api:serve_command",
        ],
    },
    include_package_data=True,
    package_data={
        "gazimed": [
            "configs/*.yaml",
            "templates/*.mni.gz",
            "data/atlases/*.nii.gz",
        ],
    },
    zip_safe=False,
    keywords=[
        "alzheimers", "dementia", "medical-imaging", "deep-learning", 
        "mri", "pet", "brain-imaging", "healthcare", "ai", "pytorch"
    ],
    project_urls={
        "Bug Reports": "https://github.com/gazimed/alzheimers-detection/issues",
        "Documentation": "https://gazimed.readthedocs.io/",
        "Source": "https://github.com/gazimed/alzheimers-detection",
        "Clinical Validation": "https://gazimed.ai/clinical-studies",
    },
)