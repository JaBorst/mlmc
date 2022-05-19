from setuptools import setup
import os
import subprocess


# Get Version from Pipeline, ignore leading 'v'
if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG'][1:]
elif os.environ.get('CI_JOB_ID'):
    version = os.environ['CI_JOB_ID']

# For local builds:
else:
    try:
        # Get latest git tag
        result = subprocess.run("git describe --tags", shell=True, stdout=subprocess.PIPE)
        version = result.stdout.decode('utf-8')[1:-1] + "-local"
    except:
        version = "local"


with open("mlmc/_version.py", "w") as f:
    f.write(f"__version__='{version}'")

setup(
    name='melmac',
    python_requires='>=3.7',
    version=version,
    packages=['mlmc',
              'mlmc.data',
              'mlmc.graph',
              'mlmc.loss',
              'mlmc.metrics',
              'mlmc.models',
              'mlmc.models.abstracts',
              'mlmc.models.standard',
              'mlmc.models.zeroshot',
              'mlmc.models.zeroshot.encoder',
              'mlmc.models.zeroshot.embedding',
              'mlmc.models.zeroshot.graph',
              'mlmc.callbacks',
              'mlmc.modules',
              'mlmc.thresholds',
              'mlmc.representation'
              ],
    url='',
    license='',
    author='Janos Borst',
    author_email='borst@informatik.uni-leipzig.de',
    description='A package specialized in neural multilabel and multiclass text classification.',
    install_requires=['transformers',
                      'nlpaug',
                      'nltk',
                      'scikit-learn',
                      'datasets',
                      'pytorch-ignite',
                      'tqdm',
                      'networkx',
                      'bs4',
                      "dill",
                      'pytest',
                      'pytest-cov',
                      'rdflib',
                      'h5py',
                      'datasketch',
                      'torch>=1.5.1',
                      'sentencepiece'
    ],
    include_package_data=True,
)
