import cv2
import matplotlib.pyplot as plt


def show_comparison(original, transformed, title):
    """显示原图与变换后的图像对比"""
    plt.figure(figsize=(10, 4))

    # 转换颜色空间（OpenCV使用BGR，matplotlib使用RGB）
    if len(original.shape) == 3:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = original
        transformed_rgb = transformed

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("原始图像")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(transformed_rgb)
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()