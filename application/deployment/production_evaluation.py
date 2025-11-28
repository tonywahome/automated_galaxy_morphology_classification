#!/usr/bin/env python3
"""
Production Model Evaluation System
Monitors and evaluates the deployed Galaxy AI model in production.
"""

import requests
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Smooth",
    "Cigar-shaped", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on No Bulge", "Edge-on With Bulge"
]


class ProductionEvaluator:
    """Evaluates model performance in production environment."""
    
    def __init__(self, api_endpoint: str, test_data_dir: str):
        self.api_endpoint = api_endpoint.rstrip('/')
        self.test_data_dir = Path(test_data_dir)
        self.metrics = defaultdict(list)
        self.predictions = []
        
    def check_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{self.api_endpoint}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✓ API is healthy: {response.json()}")
                return True
            else:
                logger.error(f"✗ API unhealthy: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"✗ API unreachable: {e}")
            return False
    
    def evaluate_single_image(self, image_path: Path, true_label: int) -> Dict:
        """Evaluate single image prediction."""
        start_time = time.time()
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (image_path.name, f, 'image/jpeg')}
                response = requests.post(
                    f"{self.api_endpoint}/predict",
                    files=files,
                    timeout=30
                )
            
            latency = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                result = response.json()
                pred = result['prediction']
                
                evaluation = {
                    'image': image_path.name,
                    'true_label': true_label,
                    'true_class': CLASS_NAMES[true_label],
                    'predicted_label': pred['class_id'],
                    'predicted_class': pred['class_name'],
                    'confidence': pred['confidence'],
                    'correct': pred['class_id'] == true_label,
                    'latency_ms': latency,
                    'api_latency_ms': result.get('latency_ms', 0),
                    'probabilities': result.get('probabilities', {}),
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"✓ {image_path.name}: {pred['class_name']} "
                          f"({pred['confidence']:.2%}) - {latency:.1f}ms")
                return evaluation
            else:
                logger.error(f"✗ Prediction failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"✗ Error evaluating {image_path.name}: {e}")
            return None
    
    def evaluate_test_set(self) -> Dict:
        """Evaluate entire test set."""
        logger.info("=" * 80)
        logger.info("PRODUCTION MODEL EVALUATION")
        logger.info("=" * 80)
        
        if not self.check_health():
            logger.error("API not healthy. Aborting evaluation.")
            return None
        
        # Collect test images
        test_images = []
        for class_id in range(10):
            class_dir = self.test_data_dir / f"class_{class_id}"
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                test_images.extend([(img, class_id) for img in images[:50]])  # Max 50 per class
        
        if not test_images:
            logger.error("No test images found!")
            return None
        
        logger.info(f"Found {len(test_images)} test images")
        
        # Evaluate each image
        results = []
        for i, (img_path, true_label) in enumerate(test_images, 1):
            logger.info(f"[{i}/{len(test_images)}] Evaluating {img_path.name}...")
            result = self.evaluate_single_image(img_path, true_label)
            if result:
                results.append(result)
            time.sleep(0.1)  # Rate limiting
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        
        # Save results
        self.save_results(results, metrics)
        
        # Generate visualizations
        self.generate_visualizations(results, metrics)
        
        return metrics
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics."""
        if not results:
            return {}
        
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        
        # Overall metrics
        metrics = {
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy': correct / total,
            'avg_confidence': np.mean([r['confidence'] for r in results]),
            'avg_latency_ms': np.mean([r['latency_ms'] for r in results]),
            'median_latency_ms': np.median([r['latency_ms'] for r in results]),
            'p95_latency_ms': np.percentile([r['latency_ms'] for r in results], 95),
            'p99_latency_ms': np.percentile([r['latency_ms'] for r in results], 99),
        }
        
        # Per-class metrics
        class_metrics = {}
        for class_id in range(10):
            class_results = [r for r in results if r['true_label'] == class_id]
            if class_results:
                class_correct = sum(1 for r in class_results if r['correct'])
                class_metrics[CLASS_NAMES[class_id]] = {
                    'total': len(class_results),
                    'correct': class_correct,
                    'accuracy': class_correct / len(class_results),
                    'avg_confidence': np.mean([r['confidence'] for r in class_results])
                }
        
        metrics['per_class'] = class_metrics
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total Predictions: {total}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"Average Confidence: {metrics['avg_confidence']:.2%}")
        logger.info(f"Average Latency: {metrics['avg_latency_ms']:.1f}ms")
        logger.info(f"P95 Latency: {metrics['p95_latency_ms']:.1f}ms")
        logger.info(f"P99 Latency: {metrics['p99_latency_ms']:.1f}ms")
        
        logger.info("\nPer-Class Performance:")
        for class_name, class_metric in class_metrics.items():
            logger.info(f"  {class_name}: {class_metric['accuracy']:.2%} "
                       f"({class_metric['correct']}/{class_metric['total']})")
        logger.info("=" * 80)
        
        return metrics
    
    def save_results(self, results: List[Dict], metrics: Dict):
        """Save evaluation results to JSON."""
        output_dir = Path("reports/production_evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'api_endpoint': self.api_endpoint,
                'metrics': metrics,
                'predictions': results
            }, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    def generate_visualizations(self, results: List[Dict], metrics: Dict):
        """Generate evaluation visualizations."""
        output_dir = Path("reports/production_evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy by class
        class_names = list(metrics['per_class'].keys())
        accuracies = [metrics['per_class'][c]['accuracy'] for c in class_names]
        
        axes[0, 0].barh(class_names, accuracies, color='steelblue')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Per-Class Accuracy in Production')
        axes[0, 0].axvline(metrics['accuracy'], color='red', linestyle='--', 
                          label=f'Overall: {metrics["accuracy"]:.2%}')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Confidence distribution
        confidences = [r['confidence'] for r in results]
        axes[0, 1].hist(confidences, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Prediction Confidence Distribution')
        axes[0, 1].axvline(metrics['avg_confidence'], color='red', linestyle='--',
                          label=f'Mean: {metrics["avg_confidence"]:.2%}')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Latency distribution
        latencies = [r['latency_ms'] for r in results]
        axes[1, 0].hist(latencies, bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Latency (ms)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Latency Distribution')
        axes[1, 0].axvline(metrics['avg_latency_ms'], color='red', linestyle='--',
                          label=f'Mean: {metrics["avg_latency_ms"]:.1f}ms')
        axes[1, 0].axvline(metrics['p95_latency_ms'], color='blue', linestyle='--',
                          label=f'P95: {metrics["p95_latency_ms"]:.1f}ms')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Confusion matrix (simplified)
        from sklearn.metrics import confusion_matrix
        y_true = [r['true_label'] for r in results]
        y_pred = [r['predicted_label'] for r in results]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                   xticklabels=range(10), yticklabels=range(10))
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('True')
        axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plot_file = output_dir / f"evaluation_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualizations saved to: {plot_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Galaxy AI model in production')
    parser.add_argument('--endpoint', required=True, help='API endpoint URL')
    parser.add_argument('--test-data', required=True, help='Test data directory')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=300, help='Monitoring interval (seconds)')
    
    args = parser.parse_args()
    
    evaluator = ProductionEvaluator(args.endpoint, args.test_data)
    
    if args.continuous:
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        while True:
            evaluator.evaluate_test_set()
            logger.info(f"Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
    else:
        evaluator.evaluate_test_set()


if __name__ == "__main__":
    main()