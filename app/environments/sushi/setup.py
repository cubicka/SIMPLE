from setuptools import setup, find_packages

setup(
    name='sushi',
    version='0.1.0',
    description='Sushi Gym Environment',
    packages=find_packages(),
    install_requires=[
        'gym>=0.9.4',
        'numpy>=1.13.0',
        'numba>=0.52.0'
    ]
)


