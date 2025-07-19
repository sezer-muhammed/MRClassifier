import argparse
import csv
import torch
from pathlib import Path
from sqlalchemy.orm import Session

from gazimed.models import HybridAlzheimersModel
from gazimed.data.database import DatabaseManager, Subject
from gazimed.data.dataset import create_data_loaders as core_create_data_loaders

def inference_all(
    ckpt_path: str,
    output_csv: str,
    target_size: tuple = (96, 109, 96),
    batch_size: int = 4,
    num_workers: int = 0,
    device: str = None
):
    # 1. Device & model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridAlzheimersModel.load_from_checkpoint(ckpt_path)
    model.to(device).eval()

    # 2. Fetch all subject_id strings from DB
    db = DatabaseManager()
    session: Session = db.get_session()
    subject_rows = session.query(Subject.subject_id).all()
    session.close()
    all_ids = [row[0] for row in subject_rows]

    # 3. Build test DataLoader over all_ids
    loaders = core_create_data_loaders(
        db_manager=db,
        train_ids=None,
        val_ids=None,
        test_ids=all_ids,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=None,
        val_transform=None,
        load_volumes=True,
        include_difference_channel=False,
        subject_filter=None,
        balanced_sampling=False,
        target_size=target_size
    )
    test_loader = loaders["test"]


    sigmoid = torch.nn.Sigmoid()
    rows = []

    with torch.no_grad():
        for batch in test_loader:
            vols  = batch["volumes"].to(device)           # ← your actual volume key
            clin  = batch["clinical_features"].to(device)
            labels = batch["alzheimer_score"].cpu().numpy()
            ids     = batch["subject_id"]

            logits = model(vols, clin)
            probs  = sigmoid(logits).cpu().squeeze().numpy()
            preds  = (probs > 0.5).astype(int)

            for sid, gt, pr, pb in zip(ids, labels, preds, probs):
                rows.append({
                    "subject_id":   sid,
                    "ground_truth": int(gt),
                    "prediction":   int(pr),
                    "probability":  float(pb),
                })


    # 5. Save to CSV
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id", "ground_truth", "prediction", "probability"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved inference results → {out_path}")

def main():
    p = argparse.ArgumentParser(
        description="Run inference on all subjects and save CSV"
    )
    p.add_argument("--ckpt",       required=True,
                   help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--output-csv", required=True,
                   help="Path to write results CSV")
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device",      type=str,
                   help="cuda|cpu (auto if unset)")
    args = p.parse_args()

    inference_all(
        ckpt_path   = args.ckpt,
        output_csv  = args.output_csv,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        device      = args.device
    )

if __name__ == "__main__":
    main()
