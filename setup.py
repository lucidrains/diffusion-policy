from setuptools import setup, find_packages

setup(
  name = 'diffusion-policy',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Diffusion Policy',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/diffusion-policy',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'robotics',
    'denoising diffusion',
    'policy network',
    'transformers'
  ],
  install_requires=[
    'accelerate',
    'beartype',
    'einops>=0.7.0',
    'ema-pytorch',
    'torch>=2.1',
    'torchvision'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
