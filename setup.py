from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

setup(
    name='arlpy',
    version='1.6',
    description='ARL Python Tools',
    long_description=readme,
    author='Mandar Chitre',
    author_email='mandar@arl.nus.edu.sg',
    url='https://github.com/org-arl/arlpy',
    license='BSD (3-clause)',
    keywords='underwater acoustics signal processing communication',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy>=1.16.2',
        'scipy>=1.2.1',
        'utm>=0.4.2',
        'pandas>=0.23.4',
        'bokeh>=1.2.0'
    ]
)
