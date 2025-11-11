import cv2
import numpy as np
import os


def read_image_safe(image_path):
    """安全读取图片，处理各种路径问题"""
    # 首先检查文件是否存在
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")

        # 尝试处理路径中的特殊字符
        # 将全角符号转换为半角符号
        corrected_path = image_path.replace('＆', '&')  # 全角＆转半角&
        if os.path.exists(corrected_path):
            print(f"找到修正后的路径: {corrected_path}")
            image_path = corrected_path
        else:
            # 尝试在常见位置查找
            filename = os.path.basename(image_path)
            possible_paths = [
                image_path,
                corrected_path,
                os.path.join(os.getcwd(), filename),
                os.path.join(os.path.dirname(os.getcwd()), filename),
                os.path.join(os.path.dirname(os.getcwd()), 'docs', filename),
                os.path.join(os.path.dirname(__file__), '..', 'docs', filename)
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    print(f"找到图片: {path}")
                    break
            else:
                print(f"错误: 无法找到图片文件: {image_path}")
                return None

    # 尝试使用不同的方式读取图片
    try:
        # 方法1: 直接读取
        img = cv2.imread(image_path)
        if img is not None:
            print(f"成功读取图片: {image_path}, 尺寸: {img.shape[1]}x{img.shape[0]}")
            return img

        # 方法2: 使用numpy读取后再转换
        print("尝试使用备用方法读取图片...")
        with open(image_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                print(f"使用备用方法成功读取图片: {image_path}")
                return img

        # 方法3: 尝试不同的颜色模式
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            print(f"使用IMREAD_UNCHANGED成功读取图片: {image_path}")
            # 如果是4通道图片（带透明度），转换为3通道
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img

        print(f"错误: 无法读取图片文件，可能格式不受支持或文件已损坏: {image_path}")
        print("支持的格式: JPG, JPEG, PNG, BMP, TIFF, WEBP等")
        return None

    except Exception as e:
        print(f"读取图片时发生错误: {e}")
        return None


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


def select_points_interactively(img, window_name="选择4个点（顺时针方向：左上、右上、右下、左下）"):
    """
    交互式选择图片中的点
    """
    if img is None:
        print("图片为空")
        return None

    display_img = img.copy()
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append([x, y])
                cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display_img, str(len(points)), (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(window_name, display_img)

                if len(points) == 4:
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
                    cv2.imshow(window_name, display_img)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("请在图片上点击选择4个点（顺时针方向：左上、右上、右下、左下）")
    print("选择完成后按任意键继续...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points if len(points) == 4 else None


def extract_frame_from_video(video_path, frame_number=0):
    """从视频中提取指定帧作为参考帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return None

    # 设置到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if ret:
        print(f"成功从视频中提取第{frame_number}帧作为参考帧")
        return frame
    else:
        print(f"无法从视频中提取第{frame_number}帧")
        return None