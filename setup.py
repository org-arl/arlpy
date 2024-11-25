from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='arlpy',
    version='1.9.1',
    description='ARL Python Tools',
    long_description=readme,
    author='Mandar Chitre',
    author_email='mandar@nus.edu.sg',
    url='https://github.com/org-arl/arlpy',
    license='BSD (3-clause)',
    keywords='underwater acoustics signal processing communication',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy>=1.24.0',
        'numpy<2.0.0',
        'scipy>=1.13.0',
        'scipy<1.14.0',
        'utm>=0.7.0',
        'pandas>=1.5.0',
        'bokeh>=3.0.0'
    ]
)
