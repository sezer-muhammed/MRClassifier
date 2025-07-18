# Implementation Plan

- [x] 1. Set up project structure and database schema





  - Create directory structure for models, data, training, evaluation, and deployment components as Python Package named gazimed
  - Design SQLite database schema for subjects table (ID, MRI path, PET path, clinical features, outcome)
  - Create model_results table for storing experiment results and metrics
  - _Requirements: 1.5, 2.6, 7.5_

- [ ] 2. Implement database management and data models
  - [x] 2.1 Create SQLite database connection and ORM models


    - Implement database connection utilities with SQLAlchemy
    - Create Subject model with MRI/PET paths, 118 clinical features, and Alzheimer's score
    - Add ModelResult model for storing experiment metrics and configurations
    - _Requirements: 7.1, 7.2_

  - [x] 2.2 Build data management utilities


    - Create functions to insert/update/query subjects from database
    - Implement data validation for file paths and clinical features
    - Add database migration and backup utilities
    - _Requirements: 1.5, 7.4_

  - [x] 2.3 Create data loading and dataset classes







    - Implement AlzheimersDataset that queries database and loads NIfTI files
    - Add data splitting utilities for train/validation/test sets
    - Create data loaders with proper batch size and cross-validation support
    - _Requirements: 4.1_


- [x] 3. Implement NIfTI preprocessing pipeline




  - [x] 3.1 Create NIfTI loader with N4 bias correction




    - Write function to load NIfTI files using nibabel
    - Implement N4 bias field correction using ANTs or SimpleITK
    - Add unit tests for NIfTI loading and bias correction
    - _Requirements: 1.1_



  - [x] 3.2 Implement MNI registration and resampling

    - Write affine registration to MNI152 template using ANTs
    - Implement resampling to 1mm³ isotropic resolution
    - Add validation for output dimensions (91, 120, 91)


    - _Requirements: 1.2, 1.3_

  - [x] 3.3 Add Z-score normalization and volume combination


    - Implement per-volume Z-score normalization
    - Create function to combine MRI and PET into 2-channel or 3-channel tensor
    - Add optional difference channel (PET - MRI) computation
    - _Requirements: 1.4, 1.5_

- [x] 4. Add medical data augmentation





  - [x] 4.1 Implement medical image augmentations


    - Create RandAffine transforms with ±10° rotation using MONAI
    - Add RandFlip, RandBiasField, and Mixup augmentations
    - Build augmentation pipeline with configurable probabilities
    - _Requirements: 3.4_




  - [x] 4.2 Create cross-validation data splitting
    - Implement 5-fold cross-validation with stratified splitting
    - Add data splitting utilities that work with database queries
    - Create PyTorch DataLoaders with proper batch size and workers
    - _Requirements: 4.1_

- [x] 5. Implement core model architecture components








  - [x] 5.1 Create 3D patch embedding module
    - Implement PatchEmbed3D with configurable patch size (4³) and embedding dimension (96)
    - Add positional encoding for 3D patches


    - Write unit tests for patch embedding dimensions and functionality
    - _Requirements: 2.1_

  - [x] 5.2 Implement Swin-UNETR encoder backbone


    - Create SwinTransformer3D with 4 stages and window size 7³
    - Implement shifted window attention mechanism for 3D data
    - Add layer normalization and residual connections
    - _Requirements: 2.2_




  - [x] 5.3 Build cross-attention fusion module
    - Implement CrossModalAttention with MRI as queries, PET as keys/values
    - Use 2 attention heads with dimension 48 as specified
    - Add attention dropout and output projection layers
    - _Requirements: 2.3, 2.4_

  - [x] 5.4 Create clinical features encoder
    - Implement MLP branch for 118 numerical clinical features
    - Add dropout layers and ReLU activations for regularization
    - Create feature normalization and scaling utilities
    - _Requirements: Design multimodal fusion_

- [ ] 6. Build PyTorch Lightning training module
  - [ ] 6.1 Implement AlzheimersLightningModule
    - Create Lightning module combining image and clinical branches
    - Implement forward pass with multimodal fusion
    - Add regression head with sigmoid activation for 0-1 output
    - _Requirements: 2.5, 2.6_

  - [ ] 6.2 Add training and validation steps
    - Implement training_step with MSE loss for regression
    - Create validation_step with metric logging
    - Add learning rate scheduling with cosine annealing
    - _Requirements: 3.1, 3.2_

  - [ ] 6.3 Configure optimizers and mixed precision
    - Set up AdamW optimizer with specified learning rate and weight decay
    - Enable FP16 mixed precision training for memory efficiency
    - Add gradient clipping and accumulation support
    - _Requirements: 3.2, 3.3_

- [ ] 7. Implement MAE-3D pretraining
  - [ ] 7.1 Create masked autoencoder for 3D volumes
    - Implement 3D patch masking with 75% mask ratio
    - Build MAE decoder for reconstruction task
    - Add reconstruction loss computation
    - _Requirements: 3.5_

  - [ ] 7.2 Add pretraining pipeline
    - Create pretraining script for 1k unlabeled T1 brains
    - Implement 300 epoch pretraining with checkpointing
    - Add transfer learning from pretrained weights to main model
    - _Requirements: 3.5_

- [ ] 8. Build evaluation and metrics system
  - [ ] 8.1 Implement regression metrics for continuous scores
    - Create metrics for AUC, sensitivity, specificity using thresholding
    - Add MSE, MAE, and correlation metrics for regression evaluation
    - Implement confidence interval calculation across folds
    - _Requirements: 4.2, 4.4_

  - [ ] 8.2 Create cross-validation evaluation pipeline
    - Implement 5-fold cross-validation with proper data splitting
    - Add statistical significance testing across folds
    - Create performance reporting with ROC and PR curves
    - _Requirements: 4.1, 4.3, 4.4_

  - [ ] 8.3 Add model comparison and benchmarking
    - Implement baseline model comparisons (simple CNN, ResNet3D)
    - Create performance comparison utilities
    - Add statistical testing for model performance differences
    - _Requirements: 4.5_

- [ ] 9. Implement explainability and interpretability
  - [ ] 9.1 Create attention visualization tools
    - Implement attention rollout for Swin transformer layers
    - Generate attention maps highlighting important brain regions
    - Add anatomical region mapping for clinical interpretation
    - _Requirements: 5.1, 5.5_

  - [ ] 9.2 Add gradient-based explanations
    - Implement Integrated Gradients for input attribution
    - Create saliency map generation for 3D volumes
    - Add quantitative metrics for explanation consistency
    - _Requirements: 5.2, 5.4_

  - [ ] 9.3 Build clinical validation tools
    - Create tools to verify attention on hippocampus and entorhinal cortex
    - Implement anatomical region overlap metrics
    - Add visualization tools for clinical review
    - _Requirements: 5.3_

- [ ] 10. Create experiment tracking and model management
  - [ ] 10.1 Integrate MLflow for experiment logging
    - Set up MLflow tracking for hyperparameters and metrics
    - Log model artifacts and checkpoints with versioning
    - Create experiment comparison and visualization tools
    - _Requirements: 7.1, 7.2_

  - [ ] 10.2 Add hyperparameter optimization
    - Implement Optuna integration for systematic parameter search
    - Define search spaces for learning rate, batch size, architecture params
    - Create automated hyperparameter tuning pipeline
    - _Requirements: 7.3_

  - [ ] 10.3 Implement model checkpointing and early stopping
    - Add ModelCheckpoint callback for best model saving
    - Implement EarlyStopping with validation AUC monitoring
    - Create model loading and resuming utilities
    - _Requirements: 3.6_

- [ ] 11. Build deployment and inference system
  - [ ] 11.1 Create model export utilities
    - Implement TorchScript export for production deployment
    - Add ONNX export option for cross-platform compatibility
    - Create model optimization for inference speed
    - _Requirements: 6.1_

  - [ ] 11.2 Build REST API for inference
    - Create FastAPI endpoints for single and batch predictions
    - Add input validation for NIfTI files and clinical features
    - Implement proper error handling and response formatting
    - _Requirements: 6.2_

  - [ ] 11.3 Add monitoring and drift detection
    - Implement data drift detection for input distributions
    - Create performance monitoring for deployed models
    - Add alerting system for model degradation
    - _Requirements: 6.4_

- [ ] 11. Implement comprehensive testing suite
  - [ ] 11.1 Create unit tests for all components
    - Write tests for preprocessing pipeline with synthetic data
    - Add model architecture tests with forward pass validation
    - Create dataset and data loading tests
    - _Requirements: 7.4_

  - [ ] 11.2 Add integration tests
    - Implement end-to-end pipeline testing
    - Create cross-validation testing with small datasets
    - Add API integration tests for deployment endpoints
    - _Requirements: 7.4_

  - [ ] 11.3 Build performance and clinical validation tests
    - Create memory profiling and speed benchmarking tests
    - Add clinical validation tests with known cases
    - Implement robustness testing across different data distributions
    - _Requirements: 4.2, 4.3_

- [ ] 12. Create training and evaluation scripts
  - [ ] 12.1 Build main training script
    - Create command-line interface for training configuration
    - Implement training loop with Lightning Trainer
    - Add logging, checkpointing, and progress monitoring
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ] 12.2 Create evaluation and inference scripts
    - Build evaluation script for trained models
    - Create inference script for new data predictions
    - Add batch processing capabilities for large datasets
    - _Requirements: 4.1, 6.5_

  - [ ] 12.3 Add configuration management
    - Create YAML configuration files for different experiments
    - Implement configuration validation and default handling
    - Add environment-specific configuration support
    - _Requirements: 7.5_