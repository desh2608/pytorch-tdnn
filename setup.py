from setuptools import setup, find_packages

long_description = open('README.md').read()

setup(
  name = 'pytorch-tdnn',
  packages = find_packages(),
  version = '1.0.0',
  license = 'Apache 2.0',
  description = 'TDNN and TDNN-F layers in PyTorch',
  long_description = long_description,
  long_description_content_type="text/markdown",
  author = 'Desh Raj',
  author_email = 'r.desh26@gmail.com',
  url = 'https://github.com/desh2608/pytorch-tdnn',
  keywords = [
    'speech recognition',
    'time delay neural networks',
    'factored TDNN',
    'acoustic modeling'
  ],
  install_requires=[
    'torch>=1.5',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)