from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tunnel_pilot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # --- 1. INSTALL LAUNCH FILES ---
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),

        # --- 2. INSTALL MODEL FILES ---
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Janith-Chamikara',
    maintainer_email='janithchamikara13@gmail.com',
    description='CNN based drone navigation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cnn_navigator = tunnel_pilot.cnn_navigator_node:main',
        ],
    },
)
