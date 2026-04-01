import sys
import os
import random
from Augmentation import augment_and_save
from Distribution import distribution
from shutil import copyfile
import shutil


TARGET = 1000


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python BalanceDataset.py <directory>")
        sys.exit(1)

    base_dir = sys.argv[1]

    class_counts = distribution(base_dir, show=False)

    for class_name in class_counts:
        class_path = os.path.join(base_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        count = len(images)
        if count > TARGET:
            images = random.sample(images, TARGET)

        
        save_dir = os.path.join("balanced", class_name)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        count_img = 0
        for img_name in images:
            if count_img == TARGET:
                break
            src = os.path.join(class_path, img_name)
            dst = os.path.join(save_dir, img_name)
            if not os.path.exists(dst):
                copyfile(src, dst)
                count_img += 1
        index = count_img
        count = round((TARGET - index) / index)
        while index < TARGET:
            img_name = random.choice(images)
            img_path = os.path.join(class_path, img_name)
            if TARGET - index < count:
                count = TARGET - index
            augment_and_save(
                img_path,
                f"aug_{index}.jpg",
                save_dir=save_dir,
                save_all=False,
                count=count
            )

            index += count
