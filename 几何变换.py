import cv2
import numpy as np
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


def get_user_input(prompt, input_type=float, default=None):
    """获取用户输入，支持默认值"""
    while True:
        try:
            user_input = input(prompt)
            if not user_input and default is not None:
                return default
            return input_type(user_input)
        except ValueError:
            print("输入无效，请重新输入！")


def get_point_input(prompt, default_points):
    """获取点坐标输入"""
    points = []
    print(prompt)
    for i, (default_x, default_y) in enumerate(default_points):
        x = get_user_input(f"点{i + 1} X坐标 (默认{default_x}): ", int, default_x)
        y = get_user_input(f"点{i + 1} Y坐标 (默认{default_y}): ", int, default_y)
        points.append([x, y])
    return np.float32(points)


def interactive_geometric_transforms(image_path):
    """交互式几何变换实现"""

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        img = create_test_image()
        print("使用测试图像")
    else:
        img = cv2.resize(img, (400, 300))  # 调整大小
        print(f"已加载图像: {image_path}")

    height, width = img.shape[:2]
    original_img = img.copy()  # 保存原图副本

    print("\n=== 交互式几何变换演示 ===")
    print("可以实时调整参数观察变换效果")

    while True:
        print("\n请选择要应用的变换:")
        print("1. 缩放变换")
        print("2. 旋转变换")
        print("3. 平移变换")
        print("4. 透视变换")
        print("5. 单应性变换 (Homography)")
        print("6. 重置图像")
        print("7. 退出程序")

        choice = input("请输入选择 (1-7): ").strip()

        if choice == '1':
            # 缩放变换
            print("\n=== 缩放变换 ===")
            print("当前图像尺寸: {}x{}".format(width, height))
            scale_x = get_user_input("请输入水平缩放因子 (默认1.5): ", float, 1.5)
            scale_y = get_user_input("请输入垂直缩放因子 (默认0.8): ", float, 0.8)

            scaled_img = cv2.resize(img, None, fx=scale_x, fy=scale_y,
                                    interpolation=cv2.INTER_LINEAR)
            show_comparison(original_img, scaled_img, f"缩放变换 (x{scale_x}, x{scale_y})")

            # 询问是否更新当前图像
            update = input("是否将缩放后的图像作为新基准? (y/n, 默认n): ").strip().lower()
            if update == 'y':
                img = scaled_img
                height, width = img.shape[:2]
                print("已更新基准图像")

        elif choice == '2':
            # 旋转变换
            print("\n=== 旋转变换 ===")
            angle = get_user_input("请输入旋转角度 (正数逆时针, 默认45): ", float, 45)
            center_x = get_user_input(f"旋转中心X (0-{width}, 默认{width // 2}): ", int, width // 2)
            center_y = get_user_input(f"旋转中心Y (0-{height}, 默认{height // 2}): ", int, height // 2)
            scale = get_user_input("缩放比例 (默认1.0): ", float, 1.0)

            center = (center_x, center_y)
            M_rotation = cv2.getRotationMatrix2D(center, angle, scale)
            rotated_img = cv2.warpAffine(img, M_rotation, (width, height))

            show_comparison(original_img, rotated_img, f"旋转变换 ({angle}°)")

            update = input("是否将旋转后的图像作为新基准? (y/n, 默认n): ").strip().lower()
            if update == 'y':
                img = rotated_img
                print("已更新基准图像")

        elif choice == '3':
            # 平移变换
            print("\n=== 平移变换 ===")
            tx = get_user_input("请输入水平平移量 (正数向右, 默认100): ", int, 100)
            ty = get_user_input("请输入垂直平移量 (正数向下, 默认50): ", int, 50)

            M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
            translated_img = cv2.warpAffine(img, M_translation, (width, height))

            show_comparison(original_img, translated_img, f"平移变换 (x+{tx}, y+{ty})")

            update = input("是否将平移后的图像作为新基准? (y/n, 默认n): ").strip().lower()
            if update == 'y':
                img = translated_img
                print("已更新基准图像")

        elif choice == '4':
            # 透视变换
            print("\n=== 透视变换 ===")
            print("原图四个角点: 左上(0,0), 右上({},0), 右下({},{}), 左下(0,{})".format(
                width - 1, width - 1, height - 1, height - 1))

            # 获取目标点坐标
            default_dst_points = [
                [50, 50],  # 左上角
                [width - 50, 100],  # 右上角
                [width - 100, height - 50],  # 右下角
                [100, height - 100]  # 左下角
            ]

            dst_points = get_point_input("请输入目标点坐标:", default_dst_points)

            # 定义源点
            src_points = np.float32([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ])

            # 应用透视变换
            M_perspective = cv2.getPerspectiveTransform(src_points, dst_points)
            perspective_img = cv2.warpPerspective(img, M_perspective, (width, height))

            show_comparison(original_img, perspective_img, "透视变换")

            update = input("是否将透视变换后的图像作为新基准? (y/n, 默认n): ").strip().lower()
            if update == 'y':
                img = perspective_img
                print("已更新基准图像")

        elif choice == '5':
            # 单应性变换 (Homography)
            print("\n=== 单应性变换 (Homography) ===")
            print("单应性变换是更通用的投影变换，可以处理平面到平面的映射")
            print("请输入4个源点和4个目标点")

            # 获取源点坐标
            default_src_points = [
                [50, 50],  # 左上角
                [width - 50, 50],  # 右上角
                [width - 50, height - 50],  # 右下角
                [50, height - 50]  # 左下角
            ]

            src_points = get_point_input("请输入源点坐标:", default_src_points)

            # 获取目标点坐标
            default_dst_points = [
                [30, 30],  # 左上角
                [width - 30, 70],  # 右上角
                [width - 70, height - 30],  # 右下角
                [70, height - 70]  # 左下角
            ]

            dst_points = get_point_input("请输入目标点坐标:", default_dst_points)

            # 计算单应性矩阵
            H, status = cv2.findHomography(src_points, dst_points)
            if H is None:
                print("错误: 无法计算单应性矩阵，点可能共线或配置无效")
                continue

            # 应用单应性变换
            homography_img = cv2.warpPerspective(img, H, (width, height))

            # 显示源点和目标点标记
            marked_img = img.copy()
            for i, pt in enumerate(src_points):
                cv2.circle(marked_img, tuple(pt.astype(int)), 6, (0, 255, 0), -1)
                cv2.putText(marked_img, f"S{i + 1}", tuple(pt.astype(int) + [10, -10]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            homography_marked = homography_img.copy()
            for i, pt in enumerate(dst_points):
                display_pt = (max(10, min(width - 50, int(pt[0]))),
                              max(30, min(height - 10, int(pt[1]))))
                cv2.circle(homography_marked, tuple(pt.astype(int)), 6, (0, 0, 255), -1)
                cv2.putText(homography_marked, f"D{i + 1}", display_pt,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 显示变换结果
            show_comparison(original_img, homography_img, "单应性变换")

            # 显示点对应关系
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
            plt.title("原图源点 (绿色)")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(homography_marked, cv2.COLOR_BGR2RGB))
            plt.title("变换后目标点 (红色)")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            # 显示单应性矩阵
            print("\n单应性矩阵 H:")
            print(H)

            update = input("是否将单应性变换后的图像作为新基准? (y/n, 默认n): ").strip().lower()
            if update == 'y':
                img = homography_img
                print("已更新基准图像")

        elif choice == '6':
            # 重置图像
            img = original_img.copy()
            height, width = img.shape[:2]
            print("图像已重置为原始状态")

        elif choice == '7':
            # 退出程序
            print("感谢使用，再见！")
            break

        else:
            print("无效选择，请重新输入！")


def create_test_image():
    """创建测试图像"""
    width, height = 400, 300
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景

    # 绘制一些几何图形
    cv2.rectangle(img, (50, 50), (350, 250), (255, 0, 0), 3)  # 蓝色矩形
    cv2.circle(img, (200, 150), 60, (0, 255, 0), -1)  # 绿色实心圆
    cv2.line(img, (100, 100), (300, 200), (0, 0, 255), 3)  # 红色对角线
    cv2.putText(img, 'OpenCV Demo', (120, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # 黑色文字
    cv2.putText(img, 'Interactive Demo', (100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # 标题

    return img


def transform_details():
    """显示变换的数学原理"""
    print("\n=== 变换矩阵原理 ===")

    # 1. 平移变换矩阵
    tx, ty = 100, 50
    M_translate = np.array([[1, 0, tx],
                            [0, 1, ty]], dtype=np.float32)
    print(f"\n1. 平移变换矩阵 (tx={tx}, ty={ty}):")
    print(M_translate)

    # 2. 旋转变换矩阵
    angle = 45
    theta = np.radians(angle)
    M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0]], dtype=np.float32)
    print(f"\n2. 旋转变换矩阵 ({angle}°):")
    print(M_rotate)

    # 3. 缩放变换矩阵
    scale_x, scale_y = 1.5, 0.8
    M_scale = np.array([[scale_x, 0, 0],
                        [0, scale_y, 0]], dtype=np.float32)
    print(f"\n3. 缩放变换矩阵 (sx={scale_x}, sy={scale_y}):")
    print(M_scale)

    # 4. 透视变换矩阵说明
    print("\n4. 透视变换使用3x3矩阵，通过4个点对计算")

    # 5. 单应性变换说明
    print("\n5. 单应性变换也使用3x3矩阵，但可以处理更一般的平面到平面映射")
    print("   单应性矩阵有8个自由度，可以通过4个点对计算")


def homography_explanation():
    """单应性变换的详细解释"""
    print("\n=== 单应性变换 (Homography) 详解 ===")
    print("单应性变换是一种投影变换，用于描述两个平面之间的映射关系。")
    print("在计算机视觉中，单应性变换常用于:")
    print("1. 图像拼接 (Image Stitching)")
    print("2. 视角校正 (Perspective Correction)")
    print("3. 增强现实 (Augmented Reality)")
    print("4. 相机标定 (Camera Calibration)")

    print("\n数学原理:")
    print("单应性矩阵 H 是一个3x3矩阵，有8个自由度:")
    print("H = [[h11, h12, h13],")
    print("     [h21, h22, h23],")
    print("     [h31, h32, h33]]")

    print("\n变换公式:")
    print("[x']   [h11 h12 h13] [x]")
    print("[y'] = [h21 h22 h23] [y]")
    print("[w']   [h31 h32 h33] [1]")

    print("\n其中 (x,y) 是源点，(x'/w', y'/w') 是目标点")
    print("至少需要4个点对才能唯一确定单应性矩阵")


if __name__ == "__main__":
    # 使用测试图像或替换为你的图像路径
    image_path = 'soallgood.jpg'  # 替换为你的图像路径

    print("=== 图像几何变换交互式演示 ===")
    print("注意: 图像将自动调整为400x300像素以便演示")

    # 显示变换原理
    transform_details()

    # 显示单应性变换详解
    homography_explanation()

    # 执行交互式几何变换
    interactive_geometric_transforms(image_path)