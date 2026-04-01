from PIL import Image, ImageOps, ImageEnhance
import random
import os
import sys

def augment_and_save(img_path, base_name, save_dir=None, save_all=True, count=5):
    

    img = Image.open(img_path)
    name, ext = os.path.splitext(base_name)
    if not save_dir:
        save_dir = os.path.dirname(img_path)

    def save(img_obj, suffix=None):
        if suffix:
            img_obj.save(os.path.join(save_dir, f"{name}_{suffix}{ext}"))
        else:
            img_obj.save(os.path.join(save_dir, f"{name}{ext}"))

    if save_all:
        save(ImageOps.mirror(img), "Flip")

        # Rotate
        save(img.rotate(45, expand=True), "Rotate")

        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        save(enhancer.enhance(random.uniform(0.5, 1.8)), "Contrast")

        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        save(enhancer.enhance(1.5), "Illumination")

        # Projective
        def find_coeffs(pa, pb):
            matrix = []
            for p1, p2 in zip(pa, pb):
                matrix.append([p1[0], p1[1], 1, 0, 0, 0,
                            -p2[0]*p1[0], -p2[0]*p1[1]])
                matrix.append([0, 0, 0, p1[0], p1[1], 1,
                            -p2[1]*p1[0], -p2[1]*p1[1]])

            A = matrix
            B = [p for point in pb for p in point]

            return list(map(float,
                __import__('numpy').linalg.lstsq(A, B, rcond=None)[0]))

        width, height = img.size
        src = [(0, 0), (width, 0), (width, height), (0, height)]

        dst = [
            (random.randint(0, 20), random.randint(0, 20)),
            (width - random.randint(0, 20), random.randint(0, 20)),
            (width - random.randint(0, 20), height - random.randint(0, 20)),
            (random.randint(0, 20), height - random.randint(0, 20))
        ]

        coeffs = find_coeffs(dst, src)

        save(img.transform(
            (width, height),
            Image.PERSPECTIVE,
            coeffs,
            Image.BICUBIC
        ), "Projective")

        # Distortion
        pixels = img.load()
        distorted = img.copy()

        for i in range(width):
            for j in range(height):
                ni = min(width - 1, max(0, i + random.randint(-2, 2)))
                nj = min(height - 1, max(0, j + random.randint(-2, 2)))
                distorted.putpixel((i, j), pixels[ni, nj])

        save(distorted, "Distortion")
    else:
        if count > 0:
            save(ImageOps.mirror(img), "Flip")
            count -= 1
        if count > 0:
            save(img.rotate(45, expand=True), "Rotate")
            count -= 1
        if count > 0:
            enhancer = ImageEnhance.Contrast(img)
            save(enhancer.enhance(random.uniform(0.5, 1.8)), "Contrast")
            count -= 1
        if count > 0:
            enhancer = ImageEnhance.Brightness(img)
            save(enhancer.enhance(1.5), "Illumination")
            count -= 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python Augmentation.py <image_path> <save_dir>")
        sys.exit(1)

    img_path = sys.argv[1]
    try:
        save_dir = sys.argv[2]
    except IndexError:
        save_dir = None

    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(img_path)

    augment_and_save(img_path, base_name, save_dir)
