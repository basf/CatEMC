import setuptools
import os

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = [] 
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

exec(open('catemc/__init__.py').read())
setuptools.setup(
    name='catemc', 
    version=__version__,
    author='Shuang Han',
    author_email='hanshuangshiren@gmail.com',
    description='A Python-based evolutionary Monte Carlo code for heterogeneous catalysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.roqs.basf.net/qm-inorganics/basf_acat',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
    ],
    install_requires=install_requires,
    entry_points={'console_scripts': ['catemc=catemc.cli:main']},
    python_requires='>=3.6',
)
