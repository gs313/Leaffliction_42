import os
import sys
import matplotlib.pyplot as plt


def distribution(base_dir=None, show=True):
    
    if not os.path.isdir(base_dir):
        print("Invalid directory")
        return

    class_counts = {}

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)

        if os.path.isdir(class_path):
            count = 0

            for file in os.listdir(class_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    count += 1

            class_counts[class_name] = count

    if not class_counts:
        print("No data found")
        return
    class_counts = dict(sorted(class_counts.items(),
                               key=lambda x: x[1], reverse=True))
    if show:
        plot_distribution(class_counts, base_dir)
    print(f"Dataset: {os.path.basename(base_dir)}")
    print(class_counts)
    return class_counts
    
def plot_distribution(class_counts, base_dir):

    labels = list(class_counts.keys())
    values = list(class_counts.values())

    dataset_name = os.path.basename(base_dir)

    plt.figure()
    plt.bar(labels, values)
    plt.title(f"{dataset_name} Dataset Distribution (Bar Chart)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title(f"{dataset_name} Dataset Distribution (Pie Chart)")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <directory>")
        sys.exit(1)
    base_dir = sys.argv[1]
    distribution(base_dir)
