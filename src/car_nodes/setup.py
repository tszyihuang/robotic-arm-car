from setuptools import find_packages, setup

setup(
    name='car_nodes', version='0.1.0', packages=find_packages(),
    package_data={'car_nodes.arm': ['LICENSE', 'servo_binding.json'],
                  'car_nodes.debug': ['camera.html'],
                  'car_nodes.vision': ['models/*/weights.pt', 'models/*/args.yaml']},
    data_files=[('share/ament_index/resource_index/packages', ['resource/car_nodes']),
                ('share/car_nodes', ['package.xml', 'tasks.txt']),
                ('share/car_nodes/launch', ['launch/car.launch.py']),
                ('share/car_nodes/config', ['config/car.yaml'])],
    install_requires=['setuptools'], zip_safe=False,
    maintainer='car maintainers', maintainer_email='maintainer@example.com',
    description='Task planner, base, arm, vision and sensor nodes.', license='Proprietary',
    entry_points={'console_scripts': [
        'plan_node = car_nodes.planner.plan:main',
        'base_node = car_nodes.base.node:main',
        'arm_node = car_nodes.arm.node:main',
        'vision_node = car_nodes.vision.node:main',
        'sensor_node = car_nodes.sensor.node:main',
        'plan = car_nodes.planner.client:main',
        'camera_debug = car_nodes.debug.camera_web:main',
        'angles_debug = car_nodes.debug.angles:main',
    ]},
)
