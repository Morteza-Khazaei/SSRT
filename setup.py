from setuptools import setup, find_packages

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()

def read_file(file):
   with open(file) as f:
        return f.read()
    
long_description = read_file('README.md')
version = read_file('VERSION')
requirements = read_requirements('requirements.txt')

setup(
    name = 'pySSRT',
    version = version,
    author = 'Morteza Khazaei',
    author_email = 'morteza.khazaei@usherbrooke.ca',
    url = 'https://github.com/Morteza-Khazaei/SSRT',
    description = 'Single Scattering Radiative Transfer model (SSRT) in Python',
    long_description_content_type = 'text/markdown',  # If this causes a warning, upgrade your setuptools package
    long_description = long_description,
    license = 'MIT license',
    package_dir={'': 'src'},
    packages = find_packages(
        where='src', 
        exclude=['dataset']
    ),  # Don't include test directory in binary distribution
    install_requires = requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]  # Update these accordingly
)