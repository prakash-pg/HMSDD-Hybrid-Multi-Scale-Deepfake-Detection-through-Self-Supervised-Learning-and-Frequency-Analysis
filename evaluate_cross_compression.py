#!/usr/bin/env python3
"""
Cross-Compression Evaluation Script
Train on FF++RAW, evaluate on C23 and C40 separately
"""

import sys
sys.path.insert(0, '/home/prakash/projects/dfd/tf217/dfd_mim')

from train_raw_test_compressed import *

def evaluate_on_dataset(model, test_root, device, config):
    """Evaluate model on a specific test dataset"""
    print(f"\n{'='*70}")
    print(f"EVALUATING ON: {test_root}")
    print(f"{'='*70}")

    # Load test dataset
    _, _, test_paths, test_labels = prepare_balanced_faceforensics_dataset(
        test_root,
        train_ratio=0.8,
        balance_strategy='undersample'
    )

    test_transform = get_memory_optimized_transforms(config['image_size'], training=False)
    test_dataset = BalancedDeepfakeDataset(test_paths, test_labels, test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Evaluate
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device).float()

            logits = model(images, training=False)
            if isinstance(logits, tuple):
                logits = logits[0]

            probs = torch.sigmoid(logits.squeeze())

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auc = roc_auc_score(all_labels, all_probs)

    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Metrics at optimal threshold
    preds_optimal = (all_probs >= optimal_threshold).astype(int)
    precision_opt = precision_score(all_labels, preds_optimal)
    recall_opt = recall_score(all_labels, preds_optimal)
    f1_opt = f1_score(all_labels, preds_optimal)
    accuracy_opt = accuracy_score(all_labels, preds_optimal)
    cm = confusion_matrix(all_labels, preds_optimal)

    # Metrics at default threshold
    preds_default = (all_probs >= 0.5).astype(int)
    precision_def = precision_score(all_labels, preds_default)
    recall_def = recall_score(all_labels, preds_default)
    f1_def = f1_score(all_labels, preds_default)
    accuracy_def = accuracy_score(all_labels, preds_default)

    return {
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': {
            'precision': precision_opt,
            'recall': recall_opt,
            'f1': f1_opt,
            'accuracy': accuracy_opt
        },
        'default_metrics': {
            'precision': precision_def,
            'recall': recall_def,
            'f1': f1_def,
            'accuracy': accuracy_def
        },
        'confusion_matrix': cm.tolist(),
        'num_samples': len(test_labels)
    }


def main():
    """Main evaluation function"""

    config = {
        'image_size': 224,
        'batch_size': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'ssl_model': 'mae',
        'use_frequency': True,
        'use_multiscale': True,
        'use_mim': True,
        'mask_ratio': 0.75,
        'mim_weight': 0.1,

        # Datasets
        'train_root': '/home/prakash/projects/dfd/tf217/dfd_mim/ff++raw',
        'test_c23_root': '/home/prakash/projects/dfd/tf217/dfd_mim/ff++c23',
        'test_c40_root': '/home/prakash/projects/dfd/tf217/dfd_mim/ff++c40',

        # Model checkpoint
        'checkpoint': './checkpoints_balanced_ffraw/best_precision_optimized_model.pth',
    }

    print("="*70)
    print("🔍 CROSS-COMPRESSION EVALUATION")
    print("="*70)
    print(f"Training dataset: {config['train_root']}")
    print(f"Test datasets: C23, C40")
    print(f"Model checkpoint: {config['checkpoint']}")
    print("="*70)

    # Load trained model
    print("\nLoading trained model...")
    model = MemoryOptimizedMIMHybridDetector(
        ssl_model_type=config['ssl_model'],
        freeze_backbone=True,
        use_frequency=config['use_frequency'],
        use_multiscale=config['use_multiscale'],
        use_mim=config['use_mim'],
        mask_ratio=config['mask_ratio'],
        mim_weight=config['mim_weight']
    ).to(config['device'])

    checkpoint = torch.load(config['checkpoint'], map_location=config['device'], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print("✓ Model loaded successfully")

    # Evaluate on C23
    print("\n" + "="*70)
    print("TEST 1: FF++RAW → FF++C23 (Moderate Compression)")
    print("="*70)
    results_c23 = evaluate_on_dataset(model, config['test_c23_root'], config['device'], config)

    print(f"\n{'='*70}")
    print("RESULTS: FF++RAW → FF++C23")
    print(f"{'='*70}")
    print(f"AUC: {results_c23['auc']*100:.2f}%")
    print(f"Optimal Threshold: {results_c23['optimal_threshold']:.3f}")
    print(f"\nOptimal Threshold Metrics:")
    print(f"  Precision: {results_c23['optimal_metrics']['precision']*100:.2f}%")
    print(f"  Recall:    {results_c23['optimal_metrics']['recall']*100:.2f}%")
    print(f"  F1-Score:  {results_c23['optimal_metrics']['f1']*100:.2f}%")
    print(f"  Accuracy:  {results_c23['optimal_metrics']['accuracy']*100:.2f}%")
    print(f"\nDefault Threshold (0.5) Metrics:")
    print(f"  Precision: {results_c23['default_metrics']['precision']*100:.2f}%")
    print(f"  Recall:    {results_c23['default_metrics']['recall']*100:.2f}%")
    print(f"  F1-Score:  {results_c23['default_metrics']['f1']*100:.2f}%")

    # Evaluate on C40
    print("\n" + "="*70)
    print("TEST 2: FF++RAW → FF++C40 (Heavy Compression)")
    print("="*70)
    results_c40 = evaluate_on_dataset(model, config['test_c40_root'], config['device'], config)

    print(f"\n{'='*70}")
    print("RESULTS: FF++RAW → FF++C40")
    print(f"{'='*70}")
    print(f"AUC: {results_c40['auc']*100:.2f}%")
    print(f"Optimal Threshold: {results_c40['optimal_threshold']:.3f}")
    print(f"\nOptimal Threshold Metrics:")
    print(f"  Precision: {results_c40['optimal_metrics']['precision']*100:.2f}%")
    print(f"  Recall:    {results_c40['optimal_metrics']['recall']*100:.2f}%")
    print(f"  F1-Score:  {results_c40['optimal_metrics']['f1']*100:.2f}%")
    print(f"  Accuracy:  {results_c40['optimal_metrics']['accuracy']*100:.2f}%")
    print(f"\nDefault Threshold (0.5) Metrics:")
    print(f"  Precision: {results_c40['default_metrics']['precision']*100:.2f}%")
    print(f"  Recall:    {results_c40['default_metrics']['recall']*100:.2f}%")
    print(f"  F1-Score:  {results_c40['default_metrics']['f1']*100:.2f}%")

    # Summary comparison
    print(f"\n{'='*70}")
    print("📊 COMPRESSION DEGRADATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Metric':<20} {'RAW→C23':<15} {'RAW→C40':<15} {'C23→C40 Δ':<15}")
    print("-"*70)
    print(f"{'AUC':<20} {results_c23['auc']*100:>6.2f}%        {results_c40['auc']*100:>6.2f}%        {(results_c23['auc']-results_c40['auc'])*100:>+6.2f}%")
    print(f"{'Recall (optimal)':<20} {results_c23['optimal_metrics']['recall']*100:>6.2f}%        {results_c40['optimal_metrics']['recall']*100:>6.2f}%        {(results_c23['optimal_metrics']['recall']-results_c40['optimal_metrics']['recall'])*100:>+6.2f}%")
    print(f"{'Precision (optimal)':<20} {results_c23['optimal_metrics']['precision']*100:>6.2f}%        {results_c40['optimal_metrics']['precision']*100:>6.2f}%        {(results_c23['optimal_metrics']['precision']-results_c40['optimal_metrics']['precision'])*100:>+6.2f}%")
    print(f"{'F1-Score (optimal)':<20} {results_c23['optimal_metrics']['f1']*100:>6.2f}%        {results_c40['optimal_metrics']['f1']*100:>6.2f}%        {(results_c23['optimal_metrics']['f1']-results_c40['optimal_metrics']['f1'])*100:>+6.2f}%")

    # Save results
    results_summary = {
        'configuration': 'Cross-Compression Evaluation: FF++RAW → C23/C40',
        'train_dataset': 'FF++RAW',
        'test_datasets': ['FF++C23', 'FF++C40'],
        'model_checkpoint': config['checkpoint'],
        'results': {
            'c23': results_c23,
            'c40': results_c40
        }
    }

    output_file = './cross_compression_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=4)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
