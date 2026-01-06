#!/usr/bin/env python

import setuptools

install_requires = [
    'numpy>=1.20.0,<2.0.0',
    'pandas>=1.3.0',
    'nibabel>=3.2.0',
    'dipy>=1.4.0',
    'scipy>=1.7.0,<2.0.0',
    'torch>=1.10.0',
    'joblib>=1.0.0',
    'tqdm>=4.62.0',
    'psutil>=5.8.0'
]

setuptools.setup(
    name='DBSIpy',
    version='1.1.0',
    description='A Python package for Diffusion Basis Spectrum Imaging (DBSI) with DTI and NODDI engines',
    url='https://github.com/kainenutt/DBSIpy',
    author='Kainen L. Utt, Jacob Blum, Y. Wang',
    author_email='k.l.utt@wustl.edu',
    license='BSD (3-Clause)',
    packages=setuptools.find_packages(exclude=("tests*",)),
    install_requires=install_requires,
    python_requires='>=3.8',
    package_data={
        'dbsipy': [
            'configs/*.ini',
            'configs/DBSI_Configs/*.ini',
            'configs/DBSI_Configs/BasisSets/*/*.csv',
        ]
    },
    entry_points={
        'console_scripts': ['DBSIpy=dbsipy.master_cli:main'],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
)

