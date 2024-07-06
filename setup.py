from setuptools import setup, find_packages

setup(
    name='forex_predictor',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train=app:train',
            'predict=app:predict'
        ],
    },
)
