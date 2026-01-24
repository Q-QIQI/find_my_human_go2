from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch RealSense camera
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'), '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true'
        }.items()
    )

    # Your human tracker node
    human_tracker_node = Node(
        package='human_tracker',
        executable='detector',
        name='human_tracker',
        output='screen'
    )

    # Image view for annotated topic
    image_view_node = Node(
        package='image_view',
        executable='image_view',
        name='image_view',
        output='screen',
        remappings=[
            ('image', '/human_tracker/annotated')
        ]
    )

    return LaunchDescription([
        realsense_launch,
        human_tracker_node,
        image_view_node
    ])