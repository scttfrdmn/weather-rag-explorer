#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements.txt for dependencies
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="weather-rag-explorer",
    version="0.1.0",
    author="Weather RAG Explorer Team",
    author_email="your.email@example.com",
    description="An AI-powered tool for exploring and analyzing historical weather data using AWS Bedrock and RAG technology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/weather-rag-explorer",
    packages=find_packages(),
    py_modules=["weather_rag_cli", "weather_data_cleaner", "check_bedrock_access", "test_faiss"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "weather-rag=weather_rag_cli:cli",
            "weather-rag-check-bedrock=check_bedrock_access:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)