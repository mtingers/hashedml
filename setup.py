from setuptools import setup
from setuptools import find_packages

setup(
    name='hashedml',
    version='0.0.3',
    description='Hash based machine learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mtingers/hashedml',
    download_url='https://pypi.python.org/pypi/hashedml',
    license='MIT',
    author='Matth Ingersoll',
    author_email='matth@mtingers.com',
    maintainer='Matth Ingersoll',
    maintainer_email='matth@mtingers.com',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        'textblob',
    ],
    entry_points={
        'console_scripts': [
            'hashedml=hashedml.hashedml:main',
        ],
    },
    packages=find_packages(exclude=['tests*']),
)
