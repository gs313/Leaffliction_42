import cv2
import os
from plantcv import plantcv as pcv
import argparse


def process_image(img_path, out_dir="out", show=True):

    pcv.params.debug = "plot" if show else None

    img, _, filename = pcv.readimage(filename=img_path)

    blur = pcv.gaussian_blur(img=img, ksize=(9, 9), sigma_x=0)

    # 3. Mask
    # แปลงเป็น HSV และทำ Binary Threshold เพื่อแยกใบไม้ออกจากพื้นหลัง
    # ... (ส่วนบนเหมือนเดิมจนถึงตอนสร้าง mask) ...
    # 3. Mask (Transformation 2)
    hsv = pcv.rgb2gray_hsv(rgb_img=blur, channel='s')
    mask = pcv.threshold.binary(gray_img=hsv, threshold=80,
                                object_type='light')

    # 4. ROI (Transformation 3)
    # สร้างขอบเขตพื้นที่สนใจ (ROI)
    roi = pcv.roi.rectangle(img=img, x=10, y=10, h=img.shape[0]-20,
                            w=img.shape[1]-20)

    # 5. กรองภาพ (แก้ Error ตรงนี้!)
    # ใน PlantCV v4 ให้ส่งแค่ mask และ roi เข้าไปตรงๆ
    filtered_mask = pcv.roi.filter(mask=mask, roi=roi,
                                   roi_type='partial')

    # 6. สร้าง Labeled Mask จาก Mask ที่กรองแล้ว
    labeled_mask, _ = pcv.create_labels(mask=filtered_mask)

    # 7. Analyze Object (Transformation 4)
    # วิเคราะห์รูปร่างจาก Labeled Mask
    shape_img = pcv.analyze.size(img=img, labeled_mask=labeled_mask)

    # 8. Pseudolandmarks (Transformation 5)
    # หาจุดพิกัดโดยใช้ Labeled Mask หรือ filtered_mask
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=img, mask=filtered_mask)

    # วาดจุด Pseudolandmarks ลงบนภาพด้วยตัวเอง (ไม่พึ่ง PlantCV outputs)
    landmark_img = img.copy()

    for pt in top:
        cv2.circle(landmark_img, (int(pt[0][0]), int(pt[0][1])),
                   5, (255, 0, 0), -1)
    for pt in bottom:
        cv2.circle(landmark_img, (int(pt[0][0]), int(pt[0][1])),
                   5, (0, 255, 0), -1)
    for pt in center_v:
        cv2.circle(landmark_img, (int(pt[0][0]), int(pt[0][1])),
                   5, (0, 0, 255), -1)

    # 9. Color Histogram (Transformation 6)
    # สร้างกราฟสี
    color_analysis, _ = pcv.visualize.histogram(img=img, mask=mask,
                                                hist_data=True,  bins=100)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        print(filename)
        filename, ext = filename.rsplit('.', 1)
        print(ext)

        pcv.print_image(img, os.path.join(out_dir, f"{filename}.{ext}"))

        pcv.print_image(blur, os.path.join(out_dir, f"{filename}_blur.{ext}"))

        pcv.print_image(mask, os.path.join(out_dir, f"{filename}_mask.{ext}"))

        pcv.print_image(filtered_mask,
                        os.path.join(out_dir, f"{filename}_roi_mask.{ext}"))

        pcv.print_image(shape_img,
                        os.path.join(out_dir, f"{filename}_analyze.{ext}"))

        pcv.print_image(landmark_img,
                        os.path.join(out_dir, f"{filename}_landmarks.{ext}"))

        histogram_path = os.path.join(out_dir, f"{filename}_histogram.png")
        color_analysis.save(histogram_path)

    print(f"✅ Saved 6 transformations for: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="single image path")
    parser.add_argument("-src", help="source folder")
    parser.add_argument("-dst", help="destination folder")

    args = parser.parse_args()

    if args.image:
        process_image(args.image, show=False)

    elif args.src and args.dst:
        os.makedirs(args.dst, exist_ok=True)

        for file in os.listdir(args.src):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(args.src, file)
                process_image(path, out_dir=args.dst, show=False)

    else:
        print("Use -h for help")


if __name__ == "__main__":
    main()
