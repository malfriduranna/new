# Action-Recognition Experiments

This repository provides runnable scripts to reproduce the experiments described in Chapter 4 of the manuscript: modality comparison, data efficiency, semantic grouping, robustness, and late fusion between an HD-GCN skeleton pipeline and a VideoMAE RGB pipeline.

## Repository Layout

- `experiments/` — experiment scripts, model adapters, plotting utilities, and training engine.
- `configs/experiments.yaml` — template configuration covering dataset paths, hyperparameters, semantic groupings, and degradation schedules.
- `configs/label_names.txt` — example label list to map class indices to human-readable names.
- `requirements.txt` — Python dependencies.

## Quickstart

1. **Install dependencies** (Python 3.10+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare manifests** with columns `clip_id,video_path,skeleton_path,label` for each split (train/val/test). Paths can be relative to the roots defined in the config.

3. **Edit `configs/experiments.yaml`** to point to your dataset roots, manifests, and checkpoints. Update `configs/label_names.txt` to reflect the dataset's class order.

4. **Run experiments** (examples assume the default config path):

   - Exp 1 (modality comparison):

     ```bash
     python -m experiments.exp1_modality --config configs/experiments.yaml
     ```

   - Exp 2 (data efficiency):

     ```bash
     python -m experiments.exp2_efficiency --config configs/experiments.yaml
     ```

   - Exp 3 (semantic grouping):

     ```bash
     python -m experiments.exp3_semantic --config configs/experiments.yaml
     ```

   - Exp 4 (robustness):

     ```bash
     python -m experiments.exp4_robustness --config configs/experiments.yaml
     ```

   - Exp 5 (late fusion):

     ```bash
     python -m experiments.exp5_fusion --config configs/experiments.yaml
     ```

## Notes

- The HD-GCN adapter is a lightweight GRU-based placeholder. Replace `experiments/models/hd_gcn_adapter.py` with a full HD-GCN implementation when available.
- VideoMAE uses Hugging Face weights (`MCG-NJU/videomae-base` by default). Point `videomae.checkpoint.init_weights` to a local fine-tuned checkpoint to avoid network downloads.
- Outputs (checkpoints, logs, plots) are written under the `output_root` defined in the config.

## Sample OZ-Football COCO Skeletons

A copy of `OZ_Football_COCO.npz` lives in `data/skeletons/`. To explode it into per-clip skeleton tensors and CSV manifests that match the default config:

1. (Optional) Drop a fresh `OZ_Football_COCO.npz` into `data/skeletons/`.
2. Run the prep script to materialize `data/skeletons/oz_football/*.npy`, write `data/manifests/{train,val,test}.csv`, create subset manifests, and update `configs/label_names.txt`:

   ```bash
   python scripts/prepare_oz_football.py
   ```

3. Create a placeholder RGB clip so the VideoMAE dataloader has something to read (replace it with real clips when available):

   ```bash
   python - <<'PY'
   import torch
   from torchvision.io import write_video
   import pathlib
   path = pathlib.Path('data/videos/dummy.mp4')
   if not path.exists():
       frames = torch.zeros((16, 64, 64, 3), dtype=torch.uint8)
       write_video(str(path), frames, fps=8)
   PY
   ```

The generated manifests already match the defaults in `configs/experiments.yaml`, so you can run the experiments immediately. The placeholder video is the same for every row; VideoMAE metrics will therefore be meaningless until you swap in actual RGB clips, but the HD-GCN skeleton branch trains as expected.

## Running on an LSF Cluster

1. Copy this repo to your cluster home (e.g., `/zhome/.../new`) and place the OZ-Football assets under `data/` as described above.
2. On a login node, bootstrap the virtualenv once:

   ```bash
   chmod +x scripts/setup_cluster_env.sh
   ./scripts/setup_cluster_env.sh
   ```

   This installs `requirements.txt` into `.venv/` and drops a sentinel file so batch jobs skip the heavy install step.
3. Submit the bundled job script (requests one GPU on `gpul40s` by default—tweak `#BSUB -q` to use another queue from `bqueues` if needed):

   ```bash
   bsub < jobs/run_all_experiments.bsub
   ```

   Runtime output streams to both `logs/%J.out` (LSF) and `logs/%J_run.log` (tee’d stream created at launch). Monitor progress with `tail -f logs/<jobid>_run.log`.

## License

MIT
