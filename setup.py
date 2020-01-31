from setuptools import setup
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    

setup(
    name='mlmc',
    version='0.0.1',
    packages=['mlmc',
              'mlmc.data',
              'mlmc.graph',
              'mlmc.layers',
              'mlmc.loss',
              'mlmc.metrics',
              'mlmc.models',
              'mlmc.representation'
              ],
    url='',
    license='',
    author='jb',
    author_email='',
    description='',
    install_requires=requirements
)
