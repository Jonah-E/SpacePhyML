from setuptools import setup, find_packages

setup(
    name='SpacePhyML',
    version='0.1.0',
    description='A Framework for distributin Machine Learning Datasets for Space and Plasma physics.',
    url='https://github.com/Jonah-E/SpacePhyML',
    author='Jonah Ekelund',
    author_email='jonahek@kth.se',
    license='MIT License',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'spacephyml=spacephyml.cli:main',  # Replace with your function
        ],
    },
    install_requires=['torch',
                      'pandas',
                      'requests',
                      'cdflib',
                      'tqdm',
                      'pyarrow',
                      'scikit-learn',
                      ],

    classifiers=[
    ],
)

