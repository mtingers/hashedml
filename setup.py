from setuptools import setup
from setuptools import find_packages                                                                

setup(
    name='hashedml',
    version='0.0.1',
    description='Hash based machine learning',
    long_description=open('README.md').read(),
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
    packages=find_packages(exclude=['tests*']),
)
