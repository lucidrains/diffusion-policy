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
    'einops>=0.6.1',
    'torch>=1.12'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
