import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='o3d_recon',
    version='1.0.0',
    author="KETI UWE members",
    author_email="moonjongsul@keti.re.kr",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UnstructuredWork",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    license='MIT',
    keywords=['Cloud Computing', 'VISION', 'AI', 'Deep Learning'],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'open3d'
    ],
)
