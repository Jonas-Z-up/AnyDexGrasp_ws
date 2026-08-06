from setuptools import setup, find_packages
import os

setup(
    name="ur_toolbox",
    version="0.1.0",
    description="Python library to control an UR robot and other equipments",
    author="Minghao Gou",
    author_email="gouminghao@gmail.com",
    url='https://github.com/graspnet/ur_toolbox',
    packages=find_packages(),
    provides=["ur_toolbox"],
    install_requires=["numpy", "math3d==3.4.1", "pyrealsense2"],
    license="GNU Lesser General Public License v3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: System :: Hardware :: Hardware Drivers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ])
