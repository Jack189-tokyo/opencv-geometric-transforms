import cv2
import numpy as np

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