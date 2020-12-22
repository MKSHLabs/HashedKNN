from setuptools import setup, find_packages


with open('README.md', 'r') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name="LSHNN",
    version="1.0.0",
    url="https://github.com/MKSHLabs/LSHNN",
    author="Ashish Kashav",
    description="Locality-sensitive hashing to implement K nearesr neighbors fast.",
    long_description_content_type="text/markdown",
    long_description=README,
    author_email="ashish.kashav1@gmail.com",
    license="MIT",
    packages=find_packages(),
    keywords=['Activity of Daily Living'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Clustering"
    ],
)

install_requires = ['numpy', 'tqdm']

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
