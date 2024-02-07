from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='GeoPredictors',
    version='0.1',
    packages=find_packages(),
    description="A python package for geological data interpretation.",
    author="Team poropermeables",
    url= 'https://github.com/ese-msc-2023/ads-arcadia-poropermeables.git',
    install_requires=required_packages,
        
)