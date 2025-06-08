from setuptools import setup, find_packages

setup(
    name='univariateviz',
    version='0.1.0',
    description='Univariate categorical data visualization using Plotly',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'plotly',
        'numpy',
        'scipy',
        'ipython'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)



