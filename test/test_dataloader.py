#!/usr/bin/env python3
"""
Comprehensive DataLoader Test Script

This script tests the complete dataloader pipeline and provides detailed statistics
and visualizations to validate the normalization and data loading process.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from gazimed.data.database import DatabaseManager
from gazimed.data.dataset import AlzheimersDataset, create_data_loaders, DataSplitter


class DataLoaderTester:
    """Comprehensive tester for the dataloader pipeline."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.results = {}
        
    def test_basic_functionality(self):
        """Test basic dataloader functionality."""
        print("=== Testing Basic DataLoader Functionality ===")
        
        # Create a small dataset for testing
        dataset = AlzheimersDataset(
            db_manager=self.db_manager,
            load_volumes=True,
            cache_volumes=False,
            include_difference_channel=False,
            validate_files=False
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            print("ERROR: Empty dataset!")
            return False
        
        # Test single sample loading
        print("Testing single sample loading...")
        sample = dataset[0]
        
        if sample is None:
            print("ERROR: Failed to load first sample!")
            return False
        
        print("Sample keys:", list(sample.keys()))
        
        # Check sample structure
        expected_keys = ['subject_id', 'alzheimer_score', 'clinical_features', 'metadata']
        if dataset.load_volumes:
            expected_keys.append('volumes')
        
        for key in expected_keys:
            if key not in sample:
                print(f"ERROR: Missing key '{key}' in sample")
                return False
        
        # Check data types and shapes
        print(f"Subject ID: {sample['subject_id']}")
        print(f"Alzheimer score: {sample['alzheimer_score']} (type: {type(sample['alzheimer_score'])})")
        print(f"Clinical features shape: {sample['clinical_features'].shape}")
        
        if 'volumes' in sample:
            print(f"Volumes shape: {sample['volumes'].shape}")
        
        print("✓ Basic functionality test passed!")
        return True
    
    def test_dataloader_batching(self):
        """Test dataloader with batching."""
        print("\n=== Testing DataLoader Batching ===")
        
        # Create dataset
        dataset = AlzheimersDataset(
            db_manager=self.db_manager,
            load_volumes=True,
            cache_volumes=False,
            include_difference_channel=False,
            validate_files=False
        )
        
        # Create dataloader with custom collate function
        from gazimed.data.dataset import custom_collate_fn
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        
        print(f"DataLoader created with {len(dataloader)} batches")
        
        # Test first batch
        try:
            batch = next(iter(dataloader))
            print("Batch keys:", list(batch.keys()))
            
            print(f"Batch size: {len(batch['subject_id'])}")
            print(f"Alzheimer scores shape: {batch['alzheimer_score'].shape}")
            print(f"Clinical features shape: {batch['clinical_features'].shape}")
            
            if 'volumes' in batch:
                print(f"Volumes shape: {batch['volumes'].shape}")
            
            print("✓ Batching test passed!")
            return True
            
        except Exception as e:
            print(f"ERROR in batching: {e}")
            return False
    
    def analyze_clinical_features(self, num_samples=100):
        """Analyze clinical features normalization."""
        print(f"\n=== Analyzing Clinical Features (n={num_samples}) ===")
        
        # Create dataset
        dataset = AlzheimersDataset(
            db_manager=self.db_manager,
            load_volumes=False,  # Only clinical features
            cache_volumes=False,
            validate_files=False
        )
        
        # Collect clinical features
        clinical_features = []
        alzheimer_scores = []
        
        sample_count = min(num_samples, len(dataset))
        
        for i in tqdm(range(sample_count), desc="Collecting clinical features"):
            sample = dataset[i]
            if sample is not None:
                clinical_features.append(sample['clinical_features'].numpy())
                alzheimer_scores.append(sample['alzheimer_score'].item())
        
        if len(clinical_features) == 0:
            print("ERROR: No valid clinical features collected!")
            return
        
        clinical_features = np.array(clinical_features)
        alzheimer_scores = np.array(alzheimer_scores)
        
        print(f"Collected {len(clinical_features)} samples")
        print(f"Clinical features shape: {clinical_features.shape}")
        
        # Calculate statistics
        stats = {
            'mean': np.mean(clinical_features),
            'std': np.std(clinical_features),
            'min': np.min(clinical_features),
            'max': np.max(clinical_features),
            'median': np.median(clinical_features),
            'q25': np.percentile(clinical_features, 25),
            'q75': np.percentile(clinical_features, 75)
        }
        
        print(f"\nClinical Features Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")
        
        # Check if values are in expected range [-1, 1] due to tanh
        in_range = np.all((clinical_features >= -1.0) & (clinical_features <= 1.0))
        print(f"All values in [-1, 1] range: {in_range}")
        
        # Store results
        self.results['clinical_features'] = {
            'stats': stats,
            'data': clinical_features,
            'scores': alzheimer_scores,
            'in_range': in_range
        }
        
        return clinical_features, alzheimer_scores
    
    def analyze_volumes(self, num_samples=20):
        """Analyze volume normalization."""
        print(f"\n=== Analyzing Volume Normalization (n={num_samples}) ===")
        
        # Create dataset with volumes
        dataset = AlzheimersDataset(
            db_manager=self.db_manager,
            load_volumes=True,
            cache_volumes=False,
            include_difference_channel=False,
            validate_files=False
        )
        
        # Collect volume statistics
        mr_stats = []
        pet_stats = []
        
        sample_count = min(num_samples, len(dataset))
        
        for i in tqdm(range(sample_count), desc="Analyzing volumes"):
            sample = dataset[i]
            if sample is not None and 'volumes' in sample:
                volumes = sample['volumes'].numpy()
                
                # MR channel (channel 0)
                mr_channel = volumes[0]
                mr_stats.append({
                    'mean': np.mean(mr_channel),
                    'std': np.std(mr_channel),
                    'min': np.min(mr_channel),
                    'max': np.max(mr_channel)
                })
                
                # PET channel (channel 1)
                pet_channel = volumes[1]
                pet_stats.append({
                    'mean': np.mean(pet_channel),
                    'std': np.std(pet_channel),
                    'min': np.min(pet_channel),
                    'max': np.max(pet_channel)
                })
        
        if len(mr_stats) == 0:
            print("ERROR: No valid volumes collected!")
            return
        
        # Calculate aggregate statistics
        mr_aggregate = {
            'mean': np.mean([s['mean'] for s in mr_stats]),
            'std': np.mean([s['std'] for s in mr_stats]),
            'min': np.min([s['min'] for s in mr_stats]),
            'max': np.max([s['max'] for s in mr_stats])
        }
        
        pet_aggregate = {
            'mean': np.mean([s['mean'] for s in pet_stats]),
            'std': np.mean([s['std'] for s in pet_stats]),
            'min': np.min([s['min'] for s in pet_stats]),
            'max': np.max([s['max'] for s in pet_stats])
        }
        
        print(f"\nMR Volume Statistics (across {len(mr_stats)} samples):")
        for key, value in mr_aggregate.items():
            print(f"  {key}: {value:.6f}")
        
        print(f"\nPET Volume Statistics (across {len(pet_stats)} samples):")
        for key, value in pet_aggregate.items():
            print(f"  {key}: {value:.6f}")
        
        # Check if values are in expected range [-1, 1] due to tanh
        mr_in_range = all(s['min'] >= -1.0 and s['max'] <= 1.0 for s in mr_stats)
        pet_in_range = all(s['min'] >= -1.0 and s['max'] <= 1.0 for s in pet_stats)
        
        print(f"\nMR values in [-1, 1] range: {mr_in_range}")
        print(f"PET values in [-1, 1] range: {pet_in_range}")
        
        # Store results
        self.results['volumes'] = {
            'mr_stats': mr_aggregate,
            'pet_stats': pet_aggregate,
            'mr_in_range': mr_in_range,
            'pet_in_range': pet_in_range
        }
        
        return mr_stats, pet_stats
    
    def create_visualizations(self, output_dir="test/results"):
        """Create comprehensive visualizations."""
        print(f"\n=== Creating Visualizations ===")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Clinical Features Analysis
        if 'clinical_features' in self.results:
            clinical_data = self.results['clinical_features']['data']
            scores = self.results['clinical_features']['scores']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Clinical Features Analysis', fontsize=16, fontweight='bold')
            
            # Overall distribution
            axes[0, 0].hist(clinical_data.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('Clinical Features Distribution')
            axes[0, 0].set_xlabel('Normalized Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Per-sample means
            sample_means = np.mean(clinical_data, axis=1)
            axes[0, 1].hist(sample_means, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Per-Sample Mean Distribution')
            axes[0, 1].set_xlabel('Sample Mean')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Feature-wise means
            feature_means = np.mean(clinical_data, axis=0)
            axes[1, 0].plot(feature_means, 'o-', markersize=2)
            axes[1, 0].set_title('Feature-wise Means (116 features)')
            axes[1, 0].set_xlabel('Feature Index')
            axes[1, 0].set_ylabel('Mean Value')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Correlation with Alzheimer's scores
            axes[1, 1].scatter(sample_means, scores, alpha=0.6)
            axes[1, 1].set_title('Clinical Features Mean vs Alzheimer Score')
            axes[1, 1].set_xlabel('Clinical Features Mean')
            axes[1, 1].set_ylabel('Alzheimer Score')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'clinical_features_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Volume Analysis (if available)
        if 'volumes' in self.results:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Volume Statistics Comparison', fontsize=16, fontweight='bold')
            
            mr_stats = self.results['volumes']['mr_stats']
            pet_stats = self.results['volumes']['pet_stats']
            
            # Statistics comparison
            stats_names = ['mean', 'std', 'min', 'max']
            mr_values = [mr_stats[stat] for stat in stats_names]
            pet_values = [pet_stats[stat] for stat in stats_names]
            
            x = np.arange(len(stats_names))
            width = 0.35
            
            axes[0].bar(x - width/2, mr_values, width, label='MR', alpha=0.7)
            axes[0].bar(x + width/2, pet_values, width, label='PET', alpha=0.7)
            axes[0].set_title('Volume Statistics Comparison')
            axes[0].set_xlabel('Statistic')
            axes[0].set_ylabel('Value')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(stats_names)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Range check visualization
            range_data = {
                'MR in [-1,1]': self.results['volumes']['mr_in_range'],
                'PET in [-1,1]': self.results['volumes']['pet_in_range']
            }
            
            colors = ['green' if v else 'red' for v in range_data.values()]
            axes[1].bar(range_data.keys(), [1 if v else 0 for v in range_data.values()], color=colors, alpha=0.7)
            axes[1].set_title('Range Validation')
            axes[1].set_ylabel('Pass (1) / Fail (0)')
            axes[1].set_ylim(0, 1.2)
            
            for i, (key, value) in enumerate(range_data.items()):
                axes[1].text(i, 0.5, 'PASS' if value else 'FAIL', ha='center', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path / 'volume_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"Visualizations saved to: {output_path}")
    
    def generate_report(self, output_dir="test/results"):
        """Generate comprehensive test report."""
        print(f"\n=== Generating Test Report ===")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive report
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'database_stats': self.db_manager.get_database_stats(),
            'test_results': self.results,
            'validation_summary': {
                'clinical_features_valid': False,
                'volumes_valid': False,
                'overall_status': 'UNKNOWN'
            }
        }
        
        # Validate clinical features
        if 'clinical_features' in self.results:
            cf_results = self.results['clinical_features']
            cf_valid = (
                cf_results['in_range'] and
                -1.0 <= cf_results['stats']['mean'] <= 1.0 and
                cf_results['stats']['std'] > 0
            )
            report['validation_summary']['clinical_features_valid'] = cf_valid
        
        # Validate volumes
        if 'volumes' in self.results:
            vol_results = self.results['volumes']
            vol_valid = (
                vol_results['mr_in_range'] and
                vol_results['pet_in_range']
            )
            report['validation_summary']['volumes_valid'] = vol_valid
        
        # Overall status
        cf_ok = report['validation_summary']['clinical_features_valid']
        vol_ok = report['validation_summary']['volumes_valid']
        
        if cf_ok and vol_ok:
            report['validation_summary']['overall_status'] = 'PASS'
        elif cf_ok or vol_ok:
            report['validation_summary']['overall_status'] = 'PARTIAL'
        else:
            report['validation_summary']['overall_status'] = 'FAIL'
        
        # Save JSON report
        with open(output_path / 'dataloader_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate text summary
        summary = f"""
DATALOADER TEST REPORT
Generated: {report['test_timestamp']}

DATABASE STATUS:
- Total subjects: {report['database_stats']['total_subjects']}
- Total results: {report['database_stats']['total_results']}

CLINICAL FEATURES VALIDATION:
"""
        
        if 'clinical_features' in self.results:
            cf_stats = self.results['clinical_features']['stats']
            summary += f"""- Status: {'PASS' if report['validation_summary']['clinical_features_valid'] else 'FAIL'}
- Mean: {cf_stats['mean']:.6f}
- Std: {cf_stats['std']:.6f}
- Range: [{cf_stats['min']:.6f}, {cf_stats['max']:.6f}]
- In [-1,1]: {self.results['clinical_features']['in_range']}
"""
        else:
            summary += "- Status: NOT TESTED\n"
        
        summary += "\nVOLUME VALIDATION:\n"
        
        if 'volumes' in self.results:
            mr_stats = self.results['volumes']['mr_stats']
            pet_stats = self.results['volumes']['pet_stats']
            summary += f"""- Status: {'PASS' if report['validation_summary']['volumes_valid'] else 'FAIL'}
- MR Range Valid: {self.results['volumes']['mr_in_range']}
- PET Range Valid: {self.results['volumes']['pet_in_range']}
- MR Stats: mean={mr_stats['mean']:.4f}, std={mr_stats['std']:.4f}
- PET Stats: mean={pet_stats['mean']:.4f}, std={pet_stats['std']:.4f}
"""
        else:
            summary += "- Status: NOT TESTED\n"
        
        summary += f"""
OVERALL STATUS: {report['validation_summary']['overall_status']}

RECOMMENDATIONS:
"""
        
        if report['validation_summary']['overall_status'] == 'PASS':
            summary += "PASS - DataLoader is ready for model training!\n"
        elif report['validation_summary']['overall_status'] == 'PARTIAL':
            summary += "WARNING - Some issues detected. Review failed validations before training.\n"
        else:
            summary += "FAIL - Critical issues detected. Fix normalization before proceeding.\n"
        
        # Save text summary
        with open(output_path / 'dataloader_test_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(summary)
        print(f"Report saved to: {output_path}")
        
        return report
    
    def run_comprehensive_test(self):
        """Run all tests and generate complete report."""
        print("Starting Comprehensive DataLoader Test")
        print("=" * 60)
        
        # Test basic functionality
        if not self.test_basic_functionality():
            print("CRITICAL: Basic functionality test failed!")
            return False
        
        # Test batching
        if not self.test_dataloader_batching():
            print("CRITICAL: Batching test failed!")
            return False
        
        # Analyze clinical features
        self.analyze_clinical_features(num_samples=50)
        
        # Analyze volumes
        self.analyze_volumes(num_samples=10)
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST COMPLETED!")
        print(f"Overall Status: {report['validation_summary']['overall_status']}")
        print("=" * 60)
        
        return report['validation_summary']['overall_status'] == 'PASS'


def main():
    """Main function to run dataloader tests."""
    tester = DataLoaderTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 DataLoader is ready for model training!")
    else:
        print("\n⚠️  Please review test results and fix issues before training.")
    
    return success


if __name__ == "__main__":
    main()