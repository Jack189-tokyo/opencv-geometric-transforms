import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os


class AdvertisementInserter:
    def __init__(self, ad_image_path):
        """
        初始化广告插入器
        """
        # 读取广告图片
        self.ad_img = cv2.imread(ad_image_path)
        if self.ad_img is None:
            raise ValueError(f"无法读取广告图片: {ad_image_path}")

        # 获取广告图片尺寸
        self.ad_h, self.ad_w = self.ad_img.shape[:2]

        # 定义广告图片的四个顶点
        self.src_points = np.float32([[0, 0], [self.ad_w - 1, 0],
                                      [self.ad_w - 1, self.ad_h - 1], [0, self.ad_h - 1]])

        # 初始化多种特征检测器
        self.orb_detector = cv2.ORB_create(nfeatures=1500)  # 增加特征点数量
        self.akaze_detector = cv2.AKAZE_create()  # 添加AKAZE检测器

        # 初始化特征匹配器
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 存储参考帧和其特征
        self.reference_frame = None
        self.ref_kp_orb = None
        self.ref_des_orb = None
        self.ref_kp_akaze = None
        self.ref_des_akaze = None

        # 存储广告牌位置
        self.ad_board_points = None

        # 跟踪状态
        self.tracking_lost_count = 0
        self.max_tracking_lost = 15  # 增加最大丢失帧数

        # 检测方法优先级
        self.detection_methods = ["orb", "akaze", "template"]  # 多种检测方法

    def set_reference_frame(self, frame, points):
        """
        设置参考帧和广告牌位置
        """
        if frame is None:
            print("错误: 参考帧为空")
            return False

        if len(points) != 4:
            print("错误: 需要4个点来设置参考帧")
            return False

        self.reference_frame = frame.copy()
        self.ad_board_points = np.float32(points)

        # 提取参考帧的多种特征
        self.ref_kp_orb, self.ref_des_orb = self.orb_detector.detectAndCompute(self.reference_frame, None)
        self.ref_kp_akaze, self.ref_des_akaze = self.akaze_detector.detectAndCompute(self.reference_frame, None)

        if (self.ref_kp_orb is None or self.ref_des_orb is None) and \
                (self.ref_kp_akaze is None or self.ref_des_akaze is None):
            print("警告: 无法从参考帧提取特征点")
            return False

        orb_count = len(self.ref_kp_orb) if self.ref_kp_orb is not None else 0
        akaze_count = len(self.ref_kp_akaze) if self.ref_kp_akaze is not None else 0

        print(f"参考帧设置完成，提取到 {orb_count} 个ORB特征点和 {akaze_count} 个AKAZE特征点")
        return True

    def auto_detect_ad_board(self, frame, method="auto"):
        """
        自动检测广告牌位置
        """
        # 检查参考帧是否设置
        if self.reference_frame is None or self.ad_board_points is None:
            print("警告: 参考帧未设置，无法进行自动检测")
            return None

        # 如果method是"auto"，则尝试所有方法
        if method == "auto":
            for detection_method in self.detection_methods:
                result = self._detect_by_method(frame, detection_method)
                if result is not None:
                    print(f"使用 {detection_method} 方法检测成功")
                    return result
            return None
        else:
            return self._detect_by_method(frame, method)

    def _detect_by_method(self, frame, method):
        """
        使用指定方法检测广告牌
        """
        if method == "orb":
            return self._detect_by_orb(frame)
        elif method == "akaze":
            return self._detect_by_akaze(frame)
        elif method == "template":
            return self._detect_by_template_matching(frame)
        else:
            return None

    def _detect_by_orb(self, frame):
        """
        使用ORB特征匹配检测广告牌位置
        """
        try:
            if self.ref_kp_orb is None or self.ref_des_orb is None:
                return None

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测当前帧的特征点
            kp_frame, des_frame = self.orb_detector.detectAndCompute(frame_gray, None)

            if des_frame is None or len(kp_frame) < 4:
                return None

            # 匹配特征点
            matches = self.bf.match(self.ref_des_orb, des_frame)

            if len(matches) < 4:
                return None

            # 按距离排序，取最佳匹配
            matches = sorted(matches, key=lambda x: x.distance)

            # 提取匹配点
            src_pts = np.float32([self.ref_kp_orb[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # 使用RANSAC计算单应性矩阵
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                return None

            # 计算内点比例 - 降低阈值到0.3
            inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0
            if inlier_ratio < 0.3:  # 降低阈值
                return None

            # 将参考帧中的广告牌顶点变换到当前帧
            if self.ad_board_points.shape != (4, 2):
                self.ad_board_points = self.ad_board_points.reshape(4, 2)

            ad_board_points = cv2.perspectiveTransform(
                self.ad_board_points.reshape(-1, 1, 2), H
            ).reshape(-1, 2)

            return np.float32(ad_board_points)
        except Exception as e:
            return None

    def _detect_by_akaze(self, frame):
        """
        使用AKAZE特征匹配检测广告牌位置
        """
        try:
            if self.ref_kp_akaze is None or self.ref_des_akaze is None:
                return None

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测当前帧的特征点
            kp_frame, des_frame = self.akaze_detector.detectAndCompute(frame_gray, None)

            if des_frame is None or len(kp_frame) < 4:
                return None

            # 匹配特征点
            matches = self.bf.match(self.ref_des_akaze, des_frame)

            if len(matches) < 4:
                return None

            # 按距离排序，取最佳匹配
            matches = sorted(matches, key=lambda x: x.distance)

            # 提取匹配点
            src_pts = np.float32([self.ref_kp_akaze[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # 使用RANSAC计算单应性矩阵
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                return None

            # 计算内点比例 - 降低阈值到0.3
            inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0
            if inlier_ratio < 0.3:  # 降低阈值
                return None

            # 将参考帧中的广告牌顶点变换到当前帧
            if self.ad_board_points.shape != (4, 2):
                self.ad_board_points = self.ad_board_points.reshape(4, 2)

            ad_board_points = cv2.perspectiveTransform(
                self.ad_board_points.reshape(-1, 1, 2), H
            ).reshape(-1, 2)

            return np.float32(ad_board_points)
        except Exception as e:
            return None

    def _detect_by_template_matching(self, frame):
        """
        使用模板匹配检测广告牌
        """
        try:
            ref_gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 使用多尺度模板匹配
            result = cv2.matchTemplate(frame_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 降低匹配阈值
            threshold = 0.5  # 降低阈值
            if max_val < threshold:
                return None

            h, w = ref_gray.shape
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            points = [
                [top_left[0], top_left[1]],
                [bottom_right[0], top_left[1]],
                [bottom_right[0], bottom_right[1]],
                [top_left[0], bottom_right[1]]
            ]

            return np.float32(points)
        except Exception as e:
            return None

    def insert_advertisement(self, frame, dst_points=None, method="auto"):
        """
        将广告插入到帧中
        """
        if frame is None:
            return None

        # 如果未提供目标点，则自动检测
        if dst_points is None:
            dst_points = self.auto_detect_ad_board(frame, method)

        # 如果自动检测失败，尝试使用之前的位置
        if dst_points is None and self.ad_board_points is not None:
            dst_points = self.ad_board_points
            self.tracking_lost_count += 1
            if self.tracking_lost_count % 5 == 0:  # 每5帧打印一次
                print(f"使用之前的位置，跟踪丢失计数: {self.tracking_lost_count}")
        elif dst_points is not None:
            self.ad_board_points = dst_points.copy()
            if self.tracking_lost_count > 0:
                print(f"跟踪恢复，之前丢失 {self.tracking_lost_count} 帧")
            self.tracking_lost_count = 0

        # 如果跟踪丢失次数过多，尝试重置
        if self.tracking_lost_count > self.max_tracking_lost:
            print("跟踪丢失过多，尝试重新检测...")
            # 尝试使用模板匹配重新检测
            dst_points = self._detect_by_template_matching(frame)
            if dst_points is not None:
                self.ad_board_points = dst_points.copy()
                self.tracking_lost_count = 0
                print("通过模板匹配重新检测成功")
            else:
                return frame

        # 如果仍然没有目标点，返回原帧
        if dst_points is None:
            return frame

        # 计算单应性矩阵
        H, status = cv2.findHomography(self.src_points, dst_points)

        if H is None:
            return frame

        # 对广告图片应用透视变换
        warped_ad = cv2.warpPerspective(
            self.ad_img, H, (frame.shape[1], frame.shape[0])
        )

        # 创建掩膜
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_points), 255)

        # 将变换后的广告图片混合到帧中
        result = frame.copy()

        # 使用加权混合，使广告看起来更自然
        alpha = 0.9  # 广告的透明度
        result[mask > 0] = cv2.addWeighted(
            result[mask > 0], 1 - alpha,
            warped_ad[mask > 0], alpha, 0
        )

        return result


def extract_frame_from_video(video_path, frame_number=0):
    """
    从视频中提取指定帧作为参考帧
    """
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

    return points


def process_video_stream(video_source, ad_image_path, reference_frame=None, ad_board_points=None):
    """
    处理视频流并实时插入广告
    """
    try:
        ad_inserter = AdvertisementInserter(ad_image_path)
    except ValueError as e:
        print(f"错误: {e}")
        return

    # 打开视频源
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"无法打开视频源: {video_source}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频源: {width}x{height}, {fps:.2f} FPS, 总帧数: {total_frames}")

    # 如果提供了参考帧，设置参考帧
    if reference_frame is not None and ad_board_points is not None:
        success = ad_inserter.set_reference_frame(reference_frame, ad_board_points)
        if not success:
            print("参考帧设置失败，将使用手动选择点模式")
    else:
        print("未提供参考帧，将使用手动选择点模式")

    # 创建窗口
    cv2.namedWindow("视频广告插入", cv2.WINDOW_NORMAL)

    # 处理帧计数器
    frame_count = 0
    start_time = time.time()

    print("开始处理视频流...")
    print("按 'q' 退出，按 'p' 暂停/继续，按 's' 保存当前帧")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("视频流结束或读取失败")
                break

            # 处理帧
            processed_frame = ad_inserter.insert_advertisement(frame, method="auto")

            if processed_frame is None:
                print("帧处理失败，跳过此帧")
                continue

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
        elif key == ord('s'):
            cv2.imwrite(f"frame_{frame_count}.jpg", processed_frame)
            print(f"帧已保存为 frame_{frame_count}.jpg")

    # 计算处理时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print(f"处理完成: {frame_count} 帧, 平均 {avg_fps:.2f} FPS")

    cap.release()
    cv2.destroyAllWindows()


def main_video():
    """
    主函数：视频流广告插入
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
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("无法从摄像头捕获帧")
            return

        reference_frame = frame
        print("已从摄像头捕获参考帧")

    elif choice == "2":
        video_path = input("请输入视频文件路径: ").strip()

        # 检查文件是否存在
        if not os.path.exists(video_path):
            print(f"文件不存在: {video_path}")
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

    if not os.path.exists(ad_image_path):
        print(f"广告图片不存在: {ad_image_path}")
        return

    # 交互式选择广告牌位置
    print("请在参考帧上选择广告牌位置")
    ad_board_points = select_points_interactively(reference_frame)

    if len(ad_board_points) != 4:
        print("需要选择4个点")
        return

    print(f"选择的广告牌位置: {ad_board_points}")

    # 处理视频流
    process_video_stream(video_source, ad_image_path, reference_frame, ad_board_points)


def main_static():
    """
    静态图片广告插入（原有功能）
    """
    times_square_path = input("请输入目标图片路径: ").strip()
    ad_image_path = input("请输入广告图片路径: ").strip()

    if not os.path.exists(times_square_path) or not os.path.exists(ad_image_path):
        print("图片文件不存在，请检查路径")
        return

    target_img = cv2.imread(times_square_path)
    if target_img is None:
        print("无法读取目标图片")
        return

    print("请在目标图片上选择广告牌位置")
    dst_points = select_points_interactively(target_img)

    if len(dst_points) != 4:
        print("需要选择4个点")
        return

    # 读取广告图片
    ad_img = cv2.imread(ad_image_path)
    if ad_img is None:
        print("无法读取广告图片")
        return

    # 计算单应性矩阵
    h, w = ad_img.shape[:2]
    src_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_points = np.float32(dst_points)

    H, status = cv2.findHomography(src_points, dst_points)

    if H is None:
        print("无法计算单应性矩阵")
        return

    # 应用变换
    warped_ad = cv2.warpPerspective(ad_img, H, (target_img.shape[1], target_img.shape[0]))

    # 创建掩膜
    mask = np.zeros((target_img.shape[0], target_img.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_points), 255)

    # 混合图像
    result = target_img.copy()
    result[mask > 0] = warped_ad[mask > 0]

    # 显示结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(ad_img, cv2.COLOR_BGR2RGB))
    plt.title("广告图片")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    plt.title("原始图片")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("插入广告后")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存结果
    cv2.imwrite("advertisement_result.jpg", result)
    print("结果已保存为 'advertisement_result.jpg'")


if __name__ == "__main__":
    print("=== 广告插入系统 ===")
    print("1. 静态图片广告插入")
    print("2. 视频流广告插入（实时特征检测）")

    choice = input("请选择模式 (1 或 2): ").strip()

    if choice == "1":
        main_static()
    elif choice == "2":
        main_video()
    else:
        print("无效选择")