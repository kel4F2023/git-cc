from setuptools import setup, find_packages

setup(
    name='git-cc',
    version='0.1.0',
    description='ML-powered conventional commit classifier',
    author='Ke Li',
    author_email='kel4@andrew.cmu.edu',
    url='https://github.com/kel4F2023/git-cc',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={
        'git_cc': ['model/classifier.joblib'],
    },
    install_requires=[
        'scikit-learn>=1.0.0',
        'joblib>=1.0.0',
        'InquirerPy>=0.3.4',
        'colorama>=0.4.6',
    ],
    entry_points={
        'console_scripts': [
            'git-cc=git_cc.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Version Control :: Git',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)