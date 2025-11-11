import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advertisement_transforms import (
    AdvertisementInserter,
    show_comparison,
    get_user_input,
    get_point_input,
    create_test_image,
    select_points_interactively,
    extract_frame_from_video,
    read_image_safe
)


def static_advertisement_demo():
    """
    静态图片广告插入演示
    """
    print("=== 静态图片广告插入演示 ===")

    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(project_root, 'docs')

    # 检查docs目录是否存在
    if os.path.exists(docs_dir):
        print(f"检测到docs目录: {docs_dir}")
        # 列出docs目录下的图片文件
        image_files = [f for f in os.listdir(docs_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        if image_files:
            print("docs目录中的图片文件:")
            for i, f in enumerate(image_files):
                print(f"  {i + 1}. {f}")

    # 获取文件路径
    target_path = input("请输入目标图片路径或文件名: ").strip()
    ad_path = input("请输入广告图片路径或文件名: ").strip()

    # 如果只输入了文件名，尝试在docs目录中查找
    if not os.path.isabs(target_path) and not os.path.dirname(target_path):
        possible_target_path = os.path.join(docs_dir, target_path)
        if os.path.exists(possible_target_path):
            target_path = possible_target_path
            print(f"使用docs目录中的目标图片: {target_path}")

    if not os.path.isabs(ad_path) and not os.path.dirname(ad_path):
        possible_ad_path = os.path.join(docs_dir, ad_path)
        if os.path.exists(possible_ad_path):
            ad_path = possible_ad_path
            print(f"使用docs目录中的广告图片: {ad_path}")

    # 安全读取图片
    target_img = read_image_safe(target_path)
    if target_img is None:
        print("无法读取目标图片，使用测试图像")
        target_img = create_test_image()

    ad_img = read_image_safe(ad_path)
    if ad_img is None:
        print("无法读取广告图片，程序退出")
        return

    # 选择广告牌位置
    print("请在目标图片上选择广告牌位置")
    dst_points = select_points_interactively(target_img)

    if dst_points is None or len(dst_points) != 4:
        print("需要选择4个点，使用默认位置")
        # 使用默认位置（图片中央）
        h, w = target_img.shape[:2]
        dst_points = [
            [w // 4, h // 4],  # 左上
            [3 * w // 4, h // 4],  # 右上
            [3 * w // 4, 3 * h // 4],  # 右下
            [w // 4, 3 * h // 4]  # 左下
        ]

    # 计算单应性矩阵
    h, w = ad_img.shape[:2]
    src_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_points = np.float32(dst_points)

    H, status = cv2.findHomography(src_points, dst_points)

    if H is None:
        print("错误: 无法计算单应性矩阵，点可能共线")
        # 尝试使用仿射变换作为备选
        print("尝试使用仿射变换...")
        src_tri = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        dst_tri = np.float32([dst_points[0], dst_points[1], dst_points[3]])
        M_affine = cv2.getAffineTransform(src_tri, dst_tri)

        warped_ad = cv2.warpAffine(ad_img, M_affine, (target_img.shape[1], target_img.shape[0]))

        # 创建近似掩膜
        mask = np.zeros((target_img.shape[0], target_img.shape[1]), dtype=np.uint8)
        roi_corners = np.array([dst_points], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, 255)
    else:
        # 应用透视变换
        warped_ad = cv2.warpPerspective(ad_img, H, (target_img.shape[1], target_img.shape[0]))

        # 创建精确掩膜
        mask = np.zeros((target_img.shape[0], target_img.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_points), 255)

    # 混合图像
    result = target_img.copy()

    # 使用更自然的混合方式
    alpha = 0.85
    result[mask > 0] = cv2.addWeighted(
        result[mask > 0], 1 - alpha,
        warped_ad[mask > 0], alpha, 0
    )

    # 显示结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    if len(ad_img.shape) == 3:
        ad_img_rgb = cv2.cvtColor(ad_img, cv2.COLOR_BGR2RGB)
    else:
        ad_img_rgb = ad_img
    plt.imshow(ad_img_rgb)
    plt.title("广告图片")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    if len(target_img.shape) == 3:
        target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    else:
        target_img_rgb = target_img
    plt.imshow(target_img_rgb)
    plt.title("原始图片")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    if len(result.shape) == 3:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    else:
        result_rgb = result
    plt.imshow(result_rgb)
    plt.title("插入广告后")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存结果
    output_path = "advertisement_result.jpg"
    cv2.imwrite(output_path, result)
    print(f"结果已保存为 '{output_path}'")


def video_advertisement_demo():
    """
    视频流广告插入演示
    """
    print("=== 视频广告插入演示 ===")
    print("1. 使用摄像头实时处理")
    print("2. 处理视频文件")

    choice = input("请选择模式 (1 或 2): ").strip()

    if choice == "1":
        camera_id = input("请输入摄像头ID (默认0): ").strip()
        camera_id = int(camera_id) if camera_id else 0
        video_source = camera_id

        # 对于摄像头，需要先捕获一帧作为参考
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            # 尝试其他摄像头
            for i in range(3):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_id = i
                    print(f"使用摄像头 {i}")
                    break

        if not cap.isOpened():
            print("无法打开任何摄像头")
            return

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("无法从摄像头捕获帧")
            return

        reference_frame = frame
        print(f"已从摄像头 {camera_id} 捕获参考帧")

    elif choice == "2":
        video_path = input("请输入视频文件路径: ").strip()

        if not os.path.exists(video_path):
            # 尝试在常见位置查找
            possible_paths = [
                video_path,
                os.path.join(os.getcwd(), video_path),
                os.path.join(os.path.dirname(os.getcwd()), video_path),
                os.path.join(os.path.dirname(__file__), '..', 'docs', os.path.basename(video_path))
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    video_path = path
                    print(f"找到视频文件: {path}")
                    break
            else:
                print(f"视频文件不存在: {video_path}")
                return

        video_source = video_path

        # 从视频中提取参考帧
        frame_number = input("请输入要提取的帧编号 (默认0): ").strip()
        frame_number = int(frame_number) if frame_number else 0

        reference_frame = extract_frame_from_video(video_path, frame_number)

        if reference_frame is None:
            print("无法从视频中提取参考帧")
            return
    else:
        print("无效选择")
        return

    # 获取广告图片路径
    ad_image_path = input("请输入广告图片路径: ").strip()

    ad_img = read_image_safe(ad_image_path)
    if ad_img is None:
        print("无法读取广告图片，程序退出")
        return

    # 交互式选择广告牌位置
    print("请在参考帧上选择广告牌位置")
    ad_board_points = select_points_interactively(reference_frame)

    if ad_board_points is None or len(ad_board_points) != 4:
        print("需要选择4个点，使用默认位置")
        # 使用默认位置
        h, w = reference_frame.shape[:2]
        ad_board_points = [
            [w // 4, h // 4],  # 左上
            [3 * w // 4, h // 4],  # 右上
            [3 * w // 4, 3 * h // 4],  # 右下
            [w // 4, 3 * h // 4]  # 左下
        ]

    print(f"选择的广告牌位置: {ad_board_points}")

    # 创建广告插入器
    try:
        # 临时保存广告图片
        temp_ad_path = "temp_ad.jpg"
        cv2.imwrite(temp_ad_path, ad_img)

        ad_inserter = AdvertisementInserter(temp_ad_path)
        success = ad_inserter.set_reference_frame(reference_frame, ad_board_points)

        # 删除临时文件
        if os.path.exists(temp_ad_path):
            os.remove(temp_ad_path)

        if not success:
            print("广告插入器初始化失败")
            return
    except Exception as e:
        print(f"创建广告插入器失败: {e}")
        return

    # 处理视频流
    process_video_stream(video_source, ad_inserter)


def process_video_stream(video_source, ad_inserter):
    """处理视频流"""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"无法打开视频源: {video_source}")

        # 如果是数字，尝试作为摄像头打开
        if isinstance(video_source, int):
            print(f"尝试打开摄像头 {video_source}...")
            cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print("无法打开视频源")
            return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频源: {width}x{height}, {fps:.2f} FPS, 总帧数: {total_frames}")

    cv2.namedWindow("视频广告插入", cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    paused = False

    print("开始处理视频流...")
    print("按 'q' 退出，按 'p' 暂停/继续，按 's' 保存当前帧")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("视频流结束或读取失败")
                break

            # 插入广告
            processed_frame = ad_inserter.insert_advertisement(frame, method="auto")

            if processed_frame is None:
                # 如果处理失败，显示原帧
                processed_frame = frame

            # 显示帧
            cv2.imshow("视频广告插入", processed_frame)

            frame_count += 1

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("暂停" if paused else "继续")
        elif key == ord('s') and not paused:
            filename = f"frame_{frame_count}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"帧已保存为 {filename}")

    # 计算处理时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print(f"处理完成: {frame_count} 帧, 平均 {avg_fps:.2f} FPS")

    cap.release()
    cv2.destroyAllWindows()


def interactive_geometric_demo():
    """交互式几何变换演示"""
    print("=== 交互式几何变换演示 ===")

    image_path = input("请输入图像路径（留空使用测试图像）: ").strip()

    if image_path:
        img = read_image_safe(image_path)
        if img is None:
            print("无法读取图像，使用测试图像")
            img = create_test_image()
        else:
            # 调整大小以便演示
            img = cv2.resize(img, (400, 300))
    else:
        img = create_test_image()
        print("使用测试图像")

    height, width = img.shape[:2]
    original_img = img.copy()

    while True:
        print("\n请选择要应用的变换:")
        print("1. 缩放变换")
        print("2. 旋转变换")
        print("3. 平移变换")
        print("4. 透视变换")
        print("5. 重置图像")
        print("6. 返回主菜单")

        choice = input("请输入选择 (1-6): ").strip()

        if choice == '1':
            # 缩放变换
            print("\n=== 缩放变换 ===")
            print(f"当前图像尺寸: {width}x{height}")
            scale_x = get_user_input("请输入水平缩放因子 (默认1.5): ", float, 1.5)
            scale_y = get_user_input("请输入垂直缩放因子 (默认0.8): ", float, 0.8)

            scaled_img = cv2.resize(img, None, fx=scale_x, fy=scale_y,
                                    interpolation=cv2.INTER_LINEAR)
            show_comparison(original_img, scaled_img, f"缩放变换 (x{scale_x}, x{scale_y})")

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
            print(f"原图四个角点: 左上(0,0), 右上({width - 1},0), 右下({width - 1},{height - 1}), 左下(0,{height - 1})")

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
            # 重置图像
            img = original_img.copy()
            height, width = img.shape[:2]
            print("图像已重置为原始状态")

        elif choice == '6':
            # 返回主菜单
            print("返回主菜单...")
            break

        else:
            print("无效选择，请重新输入！")


def main():
    """主函数 - 一键启动所有演示"""
    while True:
        print("\n" + "=" * 50)
        print("           广告插入和图像变换系统")
        print("=" * 50)
        print("1. 静态图片广告插入")
        print("2. 视频流广告插入（实时特征检测）")
        print("3. 交互式几何变换演示")
        print("4. 退出程序")
        print("=" * 50)

        choice = input("请选择模式 (1-4): ").strip()

        if choice == "1":
            static_advertisement_demo()
        elif choice == "2":
            video_advertisement_demo()
        elif choice == "3":
            interactive_geometric_demo()
        elif choice == "4":
            print("感谢使用，再见！")
            break
        else:
            print("无效选择，请重新输入！")


if __name__ == "__main__":
    main()