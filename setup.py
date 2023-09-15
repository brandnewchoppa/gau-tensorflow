from setuptools import setup, find_packages

setup(
    name = 'gau-tensorflow',
    packages = find_packages(exclude = []),
    version = '0.0.9',
    license = 'MIT',
    description = 'GAU - Transformer Quality in Linear Time - TensorFlow',
    author = 'brandnewchoppa',
    author_email = '',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/brandnewchoppa/gau-tensorflow',
    keywords = [
        'artificial intelligence',
        'deep learning',
        'transformers',
        'attention mechanism',
        'gated attention unit'
    ],
    install_requires = [
        'tensorflow>=2.13.0',
        'rope-tensorflow @ git+https://github.com/brandnewchoppa/rope-tensorflow/master#egg=v0.0.2'
    ],
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: MIT License',
        'Programming Language :: Python :: 3.11'
    ]
)
