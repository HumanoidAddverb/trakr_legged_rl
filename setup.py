from setuptools import find_packages
from distutils.core import setup

setup(
    name='trakr_legged_rl',
    version='1.0.0',
    author='Addverb Technologies',
    license="--",
    packages=find_packages(),
    author_email='humanoid@addverb.com',
    description='Vanilla RL Isaac Gym environments for Legged Robots',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib']
)
