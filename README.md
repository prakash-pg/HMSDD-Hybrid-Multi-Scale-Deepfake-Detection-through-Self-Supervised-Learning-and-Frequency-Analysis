# HMSDD: Hybrid Multi-Scale Deepfake Detection

Official PyTorch implementation of **"HMSDD: Hybrid Multi-Scale Deepfake Detection through Self-Supervised Learning and Frequency Analysis"**

**Authors**: Prakash P, Jaffino G  
**Institution**: Vellore Institute of Technology, India

## Results

| Dataset | AUC | Accuracy | Precision | Recall |
|---------|-----|----------|-----------|--------|
| FF++(C23) | 99.4% | 96.1% | 99.1% | 93.0% |
| FF++(RAW) | 99.9% | 99.6% | 99.9% | 99.2% |
| FF++(C40) | 98.0% | 91.3% | 96.4% | 85.9% |

## Key Features

- MAE-based self-supervised learning
- Cross-attention fusion (Equations 6-9)
- PR-optimized focal loss with w_pos=2.5
- 97.7% AUC cross-compression (RAW→C23)

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_ffraw_val_c23.py \
    --train_root ./ff++raw \
    --val_root ./ff++c23 \
    --batch_size 16 \
    --epochs 30
```

## Evaluation

### Cross-Dataset Evaluation
```bash
python cross_dataset_evaluation.py \
    --model_path ./checkpoints/best_model.pth \
    --test_dataset celebdf
```

### Cross-Compression Evaluation
```bash
python evaluate_cross_compression.py \
    --model_path ./checkpoints/best_model.pth \
    --compression_level c23
```

## Citation

```bibtex
@inproceedings{prakash2025hmsdd,
  title={HMSDD: Hybrid Multi-Scale Deepfake Detection through Self-Supervised Learning and Frequency Analysis},
  author={Prakash, P and Jaffino, G},
  booktitle={International Conference},
  year={2025}
}
```

## Contact

- Prakash P: prakash.p2023a@vitstudent.ac.in
- Jaffino G: jaffino.g@vit.ac.in

## License

MIT License - see LICENSE file
