# locust/locustfile.py
from locust import HttpUser, task, between
import os
import random

# Sample test images (should be in locust/test_images/)
TEST_IMAGES_DIR = "locust/test_images"

class GalaxAIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Load test images on user start."""
        self.test_images = []
        if os.path.exists(TEST_IMAGES_DIR):
            for f in os.listdir(TEST_IMAGES_DIR):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    with open(os.path.join(TEST_IMAGES_DIR, f), 'rb') as img:
                        self.test_images.append((f, img.read()))
        
        # Fallback: create dummy image if no test images
        if not self.test_images:
            from PIL import Image
            import io
            img = Image.new('RGB', (256, 256), color='black')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            self.test_images.append(('dummy.png', buf.getvalue()))
    
    @task(10)
    def predict_single(self):
        """Test single image prediction endpoint."""
        img_name, img_data = random.choice(self.test_images)
        self.client.post(
            "/predict",
            files={"image": (img_name, img_data, "image/png")}
        )
    
    @task(3)
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health")
    
    @task(2)
    def model_info(self):
        """Test model info endpoint."""
        self.client.get("/model/info")
    
    @task(1)
    def class_distribution(self):
        """Test visualization endpoint."""
        self.client.get("/visualizations/class-distribution")