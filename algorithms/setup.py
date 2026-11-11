from glob import glob
from setuptools import setup

package_name = 'algorithms'

SCRIPTS = [
    'scripts/task1a/task1A.py',
    'scripts/task1b/task1B.py',
    'scripts/task1c/task1C.py',
    'scripts/task2a/task2A_perception.py',
    'scripts/task2a/task2A_manipulation.py',
    'scripts/task2b/task2B_planning.py',
    'scripts/task2b/task2B_navigation.py',
    'scripts/task3a/task3A_perception.py',
    'scripts/task3a/task3A_manipulation.py',
    'scripts/task3a/task3A_planning.py',
    'scripts/task3a/task3A_navigation.py',
    'scripts/task3a/task3A_handover.py',
]

setup(
    name=package_name,
    version='3.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='e-Yantra',
    maintainer_email='support@e-yantra.org',
    description='Your Task solutions.',
    license='Apache-2.0',
    scripts=SCRIPTS,
)
