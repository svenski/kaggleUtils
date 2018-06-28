from setuptools import setup

setup(name='kaggleUtils',
      version='0.1',
      description='Utility functions for kaggle',
      long_description='Utility functions for kaggle competitions',
      url='http://github.com/svenski/kaggleUtils',
      author='Sergiusz Bleja',
      author_email='duckgoose@bleja.org',
      license='MIT',
      long_description=long_description,
      packages=['kaggleUtils'],
      install_requires=['pandas', 'numpy', 'sklearn'],
      keywords=['kaggle'])


      
