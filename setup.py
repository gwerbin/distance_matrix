from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read()

setup(
    name='distance-matrix',
    author='Gregory Werbin',
    author_email='outthere@me.gregwerbin.com',
    license='GPL-3.0',
    version='0.1b0',
    description='Treat a "flat" distance matrix kind of like a Numpy array',
    long_description=readme,
    packages=find_packages(exclude=['tests']),
    install_requires=requirements
)
