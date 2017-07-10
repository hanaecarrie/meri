from setuptools import setup

descr =  """
Manager d'Etudes de Reconstruction d'Images (MERI) is a package related to
parameter/outputs management and image error measurement in a context of
MRI research.
"""

long_descr = """
Manager d'Etudes de Reconstruction d'Images (MERI) is a package related to
parameter/outputs management and image error measurement in a context of MRI.
This package mainly provide: a grid searching function, a reporting class and a
error measurement module.
"""

setup(name='meri',
      version='0.0.0',
      description=descr,
      long_description=long_descr,
      classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering'],
      keywords='scientific studies mangement',
      url='https://github.com/CherkaouiHamza/meri',
      author='Cherkaoui Hamza',
      author_email='hamza.cherkao@gmail.com',
      license='MIT',
      packages=['meri'],
      install_requires=[
        'numpy>=1.12.0',
        'scipy>=0.18.0',
        'joblib>=0.10.4.dev0',
        'sklearn>=0.18.1',
        'skimage>=0.13.0',
        'humanize>=0.5.1',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False)
