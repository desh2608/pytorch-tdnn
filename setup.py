from setuptools import setup, find_packages

setup(
  name = 'pytorch-tdnn',
  packages = find_packages(),
  version = '0.1.0',
  license='Apache 2.0',
  description = 'TDNN and TDNN-F layers in PyTorch',
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