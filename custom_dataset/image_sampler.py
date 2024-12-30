import os
import random
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

class ImageSampler:
    def __init__(self, dir: str = 'raw_images', num_samples: int = 1500):
        self.dir = dir
        self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
        ])

    def sample_images(self):
        """
        Random sample and save images from the directory
        """
        img_extensions = ['jpg', 'jpeg', 'png']

        for class_name in os.listdir(self.dir):
            class_dir = os.path.join(self.dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            images_paths = self._get_valid_image_paths(class_dir, img_extensions)
            sampled_images = random.sample(images_paths, min(self.num_samples, len(images_paths)))

            output_dir = f"dataset_{self.num_samples}/{class_name}"
            os.makedirs(output_dir, exist_ok=True)

            for i, img_path in tqdm(enumerate(sampled_images), total=len(sampled_images), desc=f"Processing {class_name}"):
                self._process_and_save_image(img_path, output_dir, i)

    def _get_valid_image_paths(self, class_dir, img_extensions):
        """
        Get valid image paths
        """
        images_paths = []
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.endswith(tuple(img_extensions)):
                    img_path = os.path.join(root, file)
                    if self._is_valid_image(img_path):
                        images_paths.append(img_path)
        return images_paths

    def _is_valid_image(self, img_path):
        """
        Check if the image is valid
        """
        try:
            with Image.open(img_path) as img:
                img.verify()
            return True
        except (IOError, SyntaxError) as e:
            # print(f"Invalid image: {img_path} - {e}")
            return False

    def _process_and_save_image(self, img_path, output_dir, index):
        """
        Process and save the image
        """
        img = Image.open(img_path)
        img = self.transform(img)
        output_path = os.path.join(output_dir, f"{str(index + 1).zfill(4)}.png")
        vutils.save_image(img, output_path)