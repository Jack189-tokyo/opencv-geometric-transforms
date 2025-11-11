import cv2
import numpy as np
import time


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
        self.orb_detector = cv2.ORB_create(nfeatures=1500)
        self.akaze_detector = cv2.AKAZE_create()

        # 初始化特征匹配器
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 存储参考帧和其特征
        self.reference_frame = None
        self.ref_kp_orb = None
        self.ref_des_orb = None
        self.ref_kp_akaze = None
        self.ref_des_akaze = None
        self.ad_board_points = None

        # 跟踪状态
        self.tracking_lost_count = 0
        self.max_tracking_lost = 15

        # 检测方法优先级
        self.detection_methods = ["orb", "akaze", "template"]

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
        self.ad_board_points = np.float32(points).reshape(4, 2)

        # 提取参考帧的多种特征
        frame_gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
        self.ref_kp_orb, self.ref_des_orb = self.orb_detector.detectAndCompute(frame_gray, None)
        self.ref_kp_akaze, self.ref_des_akaze = self.akaze_detector.detectAndCompute(frame_gray, None)

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
        if self.reference_frame is None or self.ad_board_points is None:
            print("警告: 参考帧未设置，无法进行自动检测")
            return None

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
        """使用指定方法检测广告牌"""
        if method == "orb":
            return self._detect_by_orb(frame)
        elif method == "akaze":
            return self._detect_by_akaze(frame)
        elif method == "template":
            return self._detect_by_template_matching(frame)
        else:
            return None

    def _detect_by_orb(self, frame):
        """使用ORB特征匹配检测广告牌位置"""
        try:
            if self.ref_kp_orb is None or self.ref_des_orb is None:
                return None

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

            # 计算内点比例
            inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0
            if inlier_ratio < 0.3:
                return None

            # 将参考帧中的广告牌顶点变换到当前帧
            if self.ad_board_points.shape != (4, 2):
                self.ad_board_points = self.ad_board_points.reshape(4, 2)

            ad_board_points = cv2.perspectiveTransform(
                self.ad_board_points.reshape(-1, 1, 2), H
            ).reshape(-1, 2)

            return np.float32(ad_board_points)
        except Exception as e:
            print(f"ORB检测错误: {e}")
            return None

    def _detect_by_akaze(self, frame):
        """使用AKAZE特征匹配检测广告牌位置"""
        try:
            if self.ref_kp_akaze is None or self.ref_des_akaze is None:
                return None

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_frame, des_frame = self.akaze_detector.detectAndCompute(frame_gray, None)

            if des_frame is None or len(kp_frame) < 4:
                return None

            # 匹配特征点
            matches = self.bf.match(self.ref_des_akaze, des_frame)
            if len(matches) < 4:
                return None

            # 按距离排序
            matches = sorted(matches, key=lambda x: x.distance)

            # 提取匹配点
            src_pts = np.float32([self.ref_kp_akaze[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # 使用RANSAC计算单应性矩阵
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                return None

            # 计算内点比例
            inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0
            if inlier_ratio < 0.3:
                return None

            # 将参考帧中的广告牌顶点变换到当前帧
            if self.ad_board_points.shape != (4, 2):
                self.ad_board_points = self.ad_board_points.reshape(4, 2)

            ad_board_points = cv2.perspectiveTransform(
                self.ad_board_points.reshape(-1, 1, 2), H
            ).reshape(-1, 2)

            return np.float32(ad_board_points)
        except Exception as e:
            print(f"AKAZE检测错误: {e}")
            return None

    def _detect_by_template_matching(self, frame):
        """使用模板匹配检测广告牌"""
        try:
            if self.reference_frame is None:
                return None

            ref_gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 使用多尺度模板匹配
            result = cv2.matchTemplate(frame_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 匹配阈值
            threshold = 0.5
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
            print(f"模板匹配错误: {e}")
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