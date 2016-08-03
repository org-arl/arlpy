from setuptools import setup, find_packages

with open('README.rst') as f:
  readme = f.read()

with open('LICENSE') as f:
  license = f.read()

setup(
  name='arlpy',
  version='0.0.1',
  description='ARL Python Tools',
  long_description=readme,
  author='Mandar Chitre',
  author_email='mandar@arl.nus.edu.sg',
  url='https://github.com/org-arl/arlpy',
  license=license,
  packages=find_packages(exclude=('tests', 'docs'))
)
