import pytest
import os
import tempfile
import shutil
from gazimed.models.lightning_module import AlzheimersDataModule

def test_datamodule_setup_and_dataloaders(monkeypatch):
    # Create a temporary directory to simulate data_dir
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "gazimed_database.db")

    # Monkeypatch DatabaseManager to create a minimal database
    from gazimed.data.database import initialize_database
    db_manager = initialize_database(db_path)
    
    # Insert a few dummy subjects
    with db_manager.get_session() as session:
        for i in range(10):
            subject = {
                "subject_id": f"S{i:03d}",
                "mri_path": f"/tmp/mri_{i:03d}.nii.gz",
                "pet_path": f"/tmp/pet_{i:03d}.nii.gz",
                "age": 60 + i,
                "sex": "M" if i % 2 == 0 else "F",
                "alzheimer_score": 0.1 * i,
                "clinical_features": [float(j) for j in range(116)],
                "dataset_source": "TEST"
            }
            from gazimed.data.database import insert_subject
            insert_subject(session, subject)

    # Create DataModule
    dm = AlzheimersDataModule(data_dir=db_path, batch_size=2, num_workers=0)
    dm.setup()

    # Test train dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert 'clinical_features' in batch
    assert 'volumes' in batch
    assert batch['clinical_features'].shape[0] == 2

    # Test val dataloader
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    assert 'clinical_features' in batch
    assert 'volumes' in batch

    # Test test dataloader (may be None if no test split)
    test_loader = dm.test_dataloader()
    if test_loader is not None:
        batch = next(iter(test_loader))
        assert 'clinical_features' in batch
        assert 'volumes' in batch

    # Cleanup
    shutil.rmtree(temp_dir)
