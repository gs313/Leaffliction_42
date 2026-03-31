import sys
import os
from PIL import Image, ImageOps, ImageEnhance
import random


def main():
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <image_path>")
        return

    img_path = sys.argv[1]

    if not os.path.isfile(img_path):
        print("Invalid file")
        return
    dataset_root = os.path.dirname(os.path.dirname(img_path))
    relative_path = os.path.relpath(img_path, dataset_root)
    new_path = os.path.join("augmented_directory",
                            os.path.basename(dataset_root), relative_path)
    print(f"Augmented images will be saved to: {new_path}")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    base_name, ext = os.path.splitext(new_path)
    img = Image.open(img_path)

    # 1. Flip
    flip = ImageOps.mirror(img)
    flip.save(f"{base_name}_Flip{ext}")

    # 2. Rotate
    rotate = img.rotate(45, expand=True)
    rotate.save(f"{base_name}_Rotate{ext}")

    width, height = img.size
    # 3 Contrast
    enhancer = ImageEnhance.Contrast(img)
    if random.random() < 0.5:
        factor = random.uniform(0.5, 0.8)
    else:
        factor = random.uniform(1.2, 1.8)
    contrast = enhancer.enhance(factor)
    contrast.save(f"{base_name}_Contrast{ext}")

    # Illumination (Brightness)
    enhancer = ImageEnhance.Brightness(img)
    illum = enhancer.enhance(1.5)
    illum.save(f"{base_name}_Illumination{ext}")

    # 5. projective
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0,
                           -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1,
                           -p2[1]*p1[0], -p2[1]*p1[1]])

        A = matrix
        B = [p for point in pb for p in point]

        res = list(map(float,
                       __import__('numpy').linalg.lstsq(A, B, rcond=None)[0]))
        return res

    width, height = img.size

    src = [(0, 0), (width, 0), (width, height), (0, height)]

    dst = [
        (random.randint(0, 20), random.randint(0, 20)),
        (width - random.randint(0, 20), random.randint(0, 20)),
        (width - random.randint(0, 20), height - random.randint(0, 20)),
        (random.randint(0, 20), height - random.randint(0, 20))
    ]

    coeffs = find_coeffs(dst, src)

    projective = img.transform(
        (width, height),
        Image.PERSPECTIVE,
        coeffs,
        Image.BICUBIC
    )

    projective.save(f"{base_name}_Projective{ext}")

    # 6. Distortion (random pixels shift)
    pixels = img.load()
    distorted = img.copy()

    for i in range(width):
        for j in range(height):
            ni = min(width - 1, max(0, i + random.randint(-2, 2)))
            nj = min(height - 1, max(0, j + random.randint(-2, 2)))
            distorted.putpixel((i, j), pixels[ni, nj])

    distorted.save(f"{base_name}_Distortion{ext}")


if __name__ == "__main__":
    main()
