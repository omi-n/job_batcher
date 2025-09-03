from setuptools import setup
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="job-batcher",
    version="0.1.0",
    author="omi-n",
    author_email="author@example.com",
    description="A Python utility for running multiple parameter sweep jobs across multiple GPUs using tmux sessions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omi-n/job_batcher",
    # packages=find_packages(),  # Remove this since we're using py_modules
    py_modules=["job_batcher"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "job-batcher=job_batcher:main",
        ],
    },
    include_package_data=True,
)
