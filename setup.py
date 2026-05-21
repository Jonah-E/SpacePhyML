from setuptools import setup, find_packages

setup(
    name='SpacePhyML',
    version='0.1.0',
    description='A Framework for distributing Machine Learning Datasets, models and algorithms for Space and Plasma physics.',
    url='https://github.com/Jonah-E/SpacePhyML',
    author='Jonah Ekelund',
    author_email='jonahek@kth.se',
    license='MIT License',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'spacephyml=spacephyml.cli:main',
        ],
    },
    install_requires=['torch',
                      'xarray',
                      'pandas',
                      'requests',
                      'cdflib',
                      'netCDF4',
                      'tqdm',
                      'ipywidgets',
                      'pyarrow',
                      'scikit-learn',
                      ],

    classifiers=[
    ],
)

