from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

setup(
    name='arlpy',
    version='1.8.0',
    description='ARL Python Tools',
    long_description=readme,
    author='Mandar Chitre',
    author_email='mandar@nus.edu.sg',
    url='https://github.com/org-arl/arlpy',
    license='BSD (3-clause)',
    keywords='underwater acoustics signal processing communication',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy>=1.18.1',
        'scipy>=1.4.1',
        'utm>=0.5.0',
        'pandas>=1.0.1',
        'bokeh>=1.4.0'
    ]
)
