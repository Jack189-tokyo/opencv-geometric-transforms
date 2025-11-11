import numpy as np

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