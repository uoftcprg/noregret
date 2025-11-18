#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name='noregret',
    version='0.0.0.dev0',
    description='No-regret learning dynamics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/uoftcprg/noregret',
    author=(
        'Universal, Open, Free, and Transparent Computer Poker Research Group'
    ),
    author_email='juhok@cs.cmu.edu',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    keywords=[
        'artificial-intelligence',
        'game',
        'game-theory',
        'imperfect-information-game',
        'online-learning',
        'python',
    ],
    project_urls={
        'Documentation': 'https://noregret.readthedocs.io/en/latest/',
        'Source': 'https://github.com/uoftcprg/noregret',
        'Tracker': 'https://github.com/uoftcprg/noregret/issues',
    },
    packages=find_packages(),
    install_requires=[
        'numpy>=2.3.3,<3',
        'ordered-set>=4.1.0,<5',
        'scipy>=1.16.1,<2',
    ],
    python_requires='>=3.11',
)
