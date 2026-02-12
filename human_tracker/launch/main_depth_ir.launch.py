import os
from launch import LaunchDescription
from launch_ros.actions import Node
 
def generate_launch_description():
    return LaunchDescription([
        # --- Realsense 节点 ---
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='camera',
            name='camera',
            parameters=[{
                'initial_reset': True,
 
                # 1. 关闭 RGB (节省带宽)
                'enable_color': False,
 
                # 2. 开启深度 & 红外
                'enable_depth': True,
                'depth_module.profile': '848x480x15', 
                'enable_infra1': True,
                'enable_infra2': True, # 如果想夜视更好，这里其实可以设为 True
                'infra_width': 848,
                'infra_height': 480,
                'infra_fps': 15,
 
                # 3. 关闭对齐
                'align_depth.enable': False,
 
                # 4. 开启激光发射器 (黑暗中必须开启)
                'emitter_enabled': True,
                'emitter_on_off': False,
                'laser_power': 250, # 激光功率拉满
 
                # =========================================
                # 🔴 核心修改：强制注入黑暗模式参数
                # 这里直接写 Python 的 Bool 和 Int，不会报错！
                # =========================================
                'depth_module.enable_auto_exposure': False, # 关自动 (Bool)
                'depth_module.exposure': 2500,               # 设为极暗 (Int)
                'depth_module.gain': 40,                    # 最低增益 (Int)
                # =========================================
 
                # 5. 禁用其他传感器
                'enable_gyro': False,
                'enable_accel': False,
                'wait_for_device_timeout': 60.0,
                'reconnect_timeout': 10.0,
            }],
            output='screen'
        ),
 
        # --- 追踪算法节点 ---
        Node(
            package='human_tracker',
            executable='depth_tracker',
            name='depth_skeleton_tracker',
            output='screen'
        ),
 
        # --- 图像显示 ---
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            arguments=['/human_tracker/output'],
            output='screen'
        )
    ])

