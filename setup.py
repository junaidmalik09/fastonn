from setuptools import setup,find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='fastonn',
      version='0.1.1',
      description='Python library for training Operational Neural Networks (ONNs)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/junaidmalik09/fastonn',
      author='Junaid Malik',
      author_email='junaid.malik@tuni.fi',
      license='MIT',
      test_suite="tests",
      packages=find_packages(),
      package_data={'fastonn': ['fastonn/utils/data/transformation/transformation.h5']},
      include_package_data=True,
      install_requires=['python_version >= "3.4"',
                      'numpy >= 1.13',
                      'torch >= 1.3.0',
                      'torchvision >= 0.4.0',
                      'scipy >= 1.3.0',
                      'matplotlib >= 3.1.0',
                      'tqdm >= 4.40.1',
                      'pillow >= 6.2.1'
                  ],
      zip_safe=False)
