# application/locust/analyze_results.py
"""
Analyze and compare Locust load test results across different container configurations.
"""

import json
import csv
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def parse_csv_stats(file_path):
    """Parse Locust CSV stats file."""
    stats = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Name'] != 'Aggregated':  # Skip aggregated row in initial pass
                stats.append(row)
            else:
                # This is the summary row
                return {
                    'total_requests': int(row['Request Count']),
                    'failure_count': int(row['Failure Count']),
                    'avg_response_time': float(row['Average Response Time']),
                    'min_response_time': float(row['Min Response Time']),
                    'max_response_time': float(row['Max Response Time']),
                    'median_response_time': float(row['Median Response Time']),
                    'p95_response_time': float(row.get('95%', 0)) if row.get('95%') else 0,
                    'p99_response_time': float(row.get('99%', 0)) if row.get('99%') else 0,
                    'requests_per_second': float(row['Requests/s']),
                    'failures_per_second': float(row['Failures/s']),
                }
    return None

def analyze_results(results_dir):
    """Analyze all test results and generate comparison report."""
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find all test results
    test_results = {}
    for file in os.listdir(results_dir):
        if file.endswith('_stats.csv'):
            container_count = int(file.split('_')[1])
            stats_path = os.path.join(results_dir, file)
            stats = parse_csv_stats(stats_path)
            if stats:
                test_results[container_count] = stats
    
    if not test_results:
        print("No test results found!")
        return
    
    # Generate comparison report
    report_path = os.path.join(results_dir, f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GalaxyAI Load Test Comparison Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by container count
        for containers in sorted(test_results.keys()):
            stats = test_results[containers]
            success_rate = ((stats['total_requests'] - stats['failure_count']) / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
            
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Configuration: {containers} Container(s)\n")
            f.write(f"{'=' * 80}\n\n")
            
            f.write(f"Total Requests:          {stats['total_requests']:,}\n")
            f.write(f"Failed Requests:         {stats['failure_count']:,}\n")
            f.write(f"Success Rate:            {success_rate:.2f}%\n")
            f.write(f"Requests/Second:         {stats['requests_per_second']:.2f}\n")
            f.write(f"\nResponse Times (ms):\n")
            f.write(f"  Average:               {stats['avg_response_time']:.2f}\n")
            f.write(f"  Median:                {stats['median_response_time']:.2f}\n")
            f.write(f"  Min:                   {stats['min_response_time']:.2f}\n")
            f.write(f"  Max:                   {stats['max_response_time']:.2f}\n")
            f.write(f"  95th Percentile:       {stats['p95_response_time']:.2f}\n")
            f.write(f"  99th Percentile:       {stats['p99_response_time']:.2f}\n")
        
        # Comparison summary
        f.write(f"\n\n{'=' * 80}\n")
        f.write("Performance Comparison Summary\n")
        f.write(f"{'=' * 80}\n\n")
        
        f.write(f"{'Containers':<15} {'RPS':<15} {'Avg Latency':<20} {'P95 Latency':<20} {'Success Rate':<15}\n")
        f.write(f"{'-' * 80}\n")
        
        for containers in sorted(test_results.keys()):
            stats = test_results[containers]
            success_rate = ((stats['total_requests'] - stats['failure_count']) / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
            f.write(f"{containers:<15} {stats['requests_per_second']:<15.2f} {stats['avg_response_time']:<20.2f} {stats['p95_response_time']:<20.2f} {success_rate:<15.2f}%\n")
        
        # Performance gains
        if len(test_results) > 1:
            f.write(f"\n\n{'=' * 80}\n")
            f.write("Scalability Analysis\n")
            f.write(f"{'=' * 80}\n\n")
            
            baseline = test_results[sorted(test_results.keys())[0]]
            for containers in sorted(test_results.keys())[1:]:
                stats = test_results[containers]
                rps_improvement = ((stats['requests_per_second'] - baseline['requests_per_second']) / baseline['requests_per_second'] * 100)
                latency_improvement = ((baseline['avg_response_time'] - stats['avg_response_time']) / baseline['avg_response_time'] * 100)
                
                f.write(f"{containers} vs 1 container:\n")
                f.write(f"  RPS Improvement:       {rps_improvement:+.2f}%\n")
                f.write(f"  Latency Improvement:   {latency_improvement:+.2f}%\n")
                f.write(f"  Theoretical Speedup:   {containers}x\n")
                f.write(f"  Actual Speedup:        {stats['requests_per_second'] / baseline['requests_per_second']:.2f}x\n")
                f.write(f"  Scaling Efficiency:    {(stats['requests_per_second'] / baseline['requests_per_second']) / containers * 100:.2f}%\n\n")
    
    print(f"\nComparison report generated: {report_path}")
    print("\nKey Findings:")
    
    # Print summary to console
    for containers in sorted(test_results.keys()):
        stats = test_results[containers]
        print(f"\n{containers} Container(s):")
        print(f"  RPS: {stats['requests_per_second']:.2f}")
        print(f"  Avg Latency: {stats['avg_response_time']:.2f}ms")
        print(f"  P95 Latency: {stats['p95_response_time']:.2f}ms")
    
    # Generate visualization
    generate_plots(test_results, results_dir)

def generate_plots(test_results, results_dir):
    """Generate visualization plots."""
    
    containers = sorted(test_results.keys())
    rps = [test_results[c]['requests_per_second'] for c in containers]
    avg_latency = [test_results[c]['avg_response_time'] for c in containers]
    p95_latency = [test_results[c]['p95_response_time'] for c in containers]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GalaxyAI Load Test Results Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Requests per Second
    axes[0, 0].bar(containers, rps, color='steelblue')
    axes[0, 0].set_xlabel('Number of Containers')
    axes[0, 0].set_ylabel('Requests per Second')
    axes[0, 0].set_title('Throughput vs Container Count')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Average Latency
    axes[0, 1].plot(containers, avg_latency, marker='o', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Number of Containers')
    axes[0, 1].set_ylabel('Average Response Time (ms)')
    axes[0, 1].set_title('Average Latency vs Container Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: P95 Latency
    axes[1, 0].plot(containers, p95_latency, marker='s', linewidth=2, markersize=8, color='red')
    axes[1, 0].set_xlabel('Number of Containers')
    axes[1, 0].set_ylabel('P95 Response Time (ms)')
    axes[1, 0].set_title('P95 Latency vs Container Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scaling Efficiency
    if len(containers) > 1:
        baseline_rps = rps[0]
        scaling_efficiency = [(r / baseline_rps) / c * 100 for r, c in zip(rps, containers)]
        axes[1, 1].bar(containers, scaling_efficiency, color='green')
        axes[1, 1].axhline(y=100, color='r', linestyle='--', label='Linear Scaling')
        axes[1, 1].set_xlabel('Number of Containers')
        axes[1, 1].set_ylabel('Scaling Efficiency (%)')
        axes[1, 1].set_title('Scaling Efficiency')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"comparison_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "application/locust/results"
    
    analyze_results(results_dir)