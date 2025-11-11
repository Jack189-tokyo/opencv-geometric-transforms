"""
广告变换工具包 - 广告插入和图像变换功能
"""

from .advertisement_inserter import AdvertisementInserter
from .core import show_comparison
from .utils import (
    get_user_input,
    get_point_input,
    create_test_image,
    select_points_interactively,
    extract_frame_from_video,
    read_image_safe
)

__version__ = "1.0.0"
__all__ = [
    'AdvertisementInserter',
    'show_comparison',
    'get_user_input',
    'get_point_input',
    'create_test_image',
    'select_points_interactively',
    'extract_frame_from_video',
    'read_image_safe'
]