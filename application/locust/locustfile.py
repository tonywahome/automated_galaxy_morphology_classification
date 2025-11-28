# application/locust/locustfile.py
"""
Locust Load Testing for GalaxyAI API

This file provides comprehensive load testing scenarios for the GalaxyAI API,
including flood simulation, latency measurements, and multi-container comparisons.

Usage:
  # Standard load test
  locust -f locustfile.py --host=http://localhost:8000

  # Headless mode with custom users/spawn rate
  locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 5m --headless

  # HTML report generation
  locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 5 --run-time 3m --headless --html=results/report.html --csv=results/stats
"""

from locust import HttpUser, task, between, events, LoadTestShape
import os
import random
import time
import json
from datetime import datetime
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEST_IMAGES_DIR = "application/locust/test_images"
RESULTS_DIR = "application/locust/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global stats collection
request_stats = []


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Collect detailed request statistics."""
    request_stats.append({
        "timestamp": datetime.now().isoformat(),
        "request_type": request_type,
        "name": name,
        "response_time": response_time,
        "response_length": response_length,
        "exception": str(exception) if exception else None,
        "success": exception is None
    })


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Save detailed statistics when test ends."""
    if request_stats:
        stats_file = os.path.join(RESULTS_DIR, f"detailed_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(stats_file, 'w') as f:
            json.dump(request_stats, f, indent=2)
        logger.info(f"Detailed statistics saved to {stats_file}")
        
        # Calculate summary statistics
        successful_requests = [r for r in request_stats if r['success']]
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            summary = {
                "total_requests": len(request_stats),
                "successful_requests": len(successful_requests),
                "failed_requests": len(request_stats) - len(successful_requests),
                "success_rate": len(successful_requests) / len(request_stats) * 100,
                "avg_response_time_ms": sum(response_times) / len(response_times),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "median_response_time_ms": sorted(response_times)[len(response_times) // 2],
                "p95_response_time_ms": sorted(response_times)[int(len(response_times) * 0.95)],
                "p99_response_time_ms": sorted(response_times)[int(len(response_times) * 0.99)],
            }
            
            summary_file = os.path.join(RESULTS_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary statistics saved to {summary_file}")
            logger.info(f"Success Rate: {summary['success_rate']:.2f}%")
            logger.info(f"Avg Response Time: {summary['avg_response_time_ms']:.2f}ms")
            logger.info(f"P95 Response Time: {summary['p95_response_time_ms']:.2f}ms")


class GalaxAIUser(HttpUser):
    """
    Standard user behavior for GalaxyAI API testing.
    Simulates realistic usage patterns with varying wait times.
    """
    wait_time = between(1, 3)
    
    def on_start(self):
        """Load test images on user start."""
        self.test_images = []
        
        # Try to load from test images directory
        if os.path.exists(TEST_IMAGES_DIR):
            for f in os.listdir(TEST_IMAGES_DIR):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(TEST_IMAGES_DIR, f)
                    with open(img_path, 'rb') as img:
                        self.test_images.append((f, img.read()))
            logger.info(f"Loaded {len(self.test_images)} test images from {TEST_IMAGES_DIR}")
        
        # Try to load from data/test directory
        if not self.test_images:
            data_test_dir = "data/test"
            if os.path.exists(data_test_dir):
                for root, dirs, files in os.walk(data_test_dir):
                    for f in files[:10]:  # Limit to 10 images per class
                        if f.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(root, f)
                            with open(img_path, 'rb') as img:
                                self.test_images.append((f, img.read()))
                logger.info(f"Loaded {len(self.test_images)} test images from {data_test_dir}")
        
        # Fallback: generate synthetic test images
        if not self.test_images:
            logger.warning("No test images found, generating synthetic images")
            for i in range(5):
                img = Image.new('RGB', (256, 256), color=(random.randint(0, 255), 
                                                          random.randint(0, 255), 
                                                          random.randint(0, 255)))
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                self.test_images.append((f'synthetic_{i}.png', buf.getvalue()))
    
    @task(10)
    def predict_single(self):
        """Test single image prediction endpoint (primary workload)."""
        if not self.test_images:
            return
        
        img_name, img_data = random.choice(self.test_images)
        with self.client.post(
            "/predict",
            files={"image": (img_name, img_data, "image/png")},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(3)
    def health_check(self):
        """Test health endpoint (monitoring workload)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get('status') == 'healthy':
                        response.success()
                    else:
                        response.failure("API not healthy")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def model_info(self):
        """Test model info endpoint."""
        with self.client.get("/model/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def class_distribution(self):
        """Test visualization endpoint (low priority workload)."""
        with self.client.get("/visualizations/class-distribution", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


class GalaxAIHeavyUser(HttpUser):
    """
    Heavy user behavior for stress testing.
    Sends rapid requests with minimal wait time.
    """
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        """Load test images on user start."""
        self.test_images = []
        if os.path.exists(TEST_IMAGES_DIR):
            for f in os.listdir(TEST_IMAGES_DIR):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    with open(os.path.join(TEST_IMAGES_DIR, f), 'rb') as img:
                        self.test_images.append((f, img.read()))
        
        if not self.test_images:
            img = Image.new('RGB', (256, 256), color='red')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            self.test_images.append(('heavy_test.png', buf.getvalue()))
    
    @task(20)
    def rapid_predictions(self):
        """Rapid-fire prediction requests."""
        img_name, img_data = random.choice(self.test_images)
        self.client.post("/predict", files={"image": (img_name, img_data, "image/png")})
    
    @task(1)
    def health_check(self):
        """Occasional health check."""
        self.client.get("/health")


class FloodLoadShape(LoadTestShape):
    """
    Custom load shape for flood testing.
    
    Simulates traffic spikes and valleys:
    - Stage 1 (0-60s): Ramp up from 0 to 50 users (warm-up)
    - Stage 2 (60-120s): Maintain 50 users (baseline)
    - Stage 3 (120-180s): Spike to 150 users (peak load)
    - Stage 4 (180-240s): Drop to 30 users (cool-down)
    - Stage 5 (240-300s): Final spike to 200 users (stress test)
    """
    
    stages = [
        {"duration": 60, "users": 50, "spawn_rate": 2},    # Warm-up
        {"duration": 120, "users": 50, "spawn_rate": 1},   # Baseline
        {"duration": 180, "users": 150, "spawn_rate": 5},  # Peak load
        {"duration": 240, "users": 30, "spawn_rate": 3},   # Cool-down
        {"duration": 300, "users": 200, "spawn_rate": 10}, # Stress test
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        
        return None  # Stop test after all stages


class StepLoadShape(LoadTestShape):
    """
    Step load shape for gradual scaling tests.
    Useful for identifying performance degradation points.
    """
    
    step_time = 60  # Duration of each step in seconds
    step_load = 20  # User increase per step
    spawn_rate = 5
    time_limit = 600  # Total test duration
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = int(run_time / self.step_time)
        return (current_step * self.step_load, self.spawn_rate)