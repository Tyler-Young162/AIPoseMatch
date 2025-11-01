"""
Setup script for AI Pose Match.
Simple package metadata for installation convenience.
"""
from setuptools import setup, find_packages

setup(
    name="ai-pose-match",
    version="1.0.0",
    description="Real-time human pose detection and matting system",
    author="AI Pose Match Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
        "pillow>=10.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

