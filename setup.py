from setuptools import setup

setup(name='nlmk',
      version='1.1',
      description='NLMK fraction classifier',
      author='CeladoAI',
      author_email='r.ibatullin@celado-media.ru',
      packages=['nlmk'],
      scripts=['bin/make_clips'],
      install_requires=[
          'numpy',
          'opencv-python',
          'torch',
          'torchvision'
      ],
      zip_safe=False)
