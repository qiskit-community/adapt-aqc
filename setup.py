from setuptools import find_packages, setup

setup(
    name="quantum-isl",
    version="1.0.2",
    author="Abhishek Agarwal, Ben Jaderberg",
    author_email="abhishek.agarwal@npl.co.uk, benjamin.jaderberg@ibm.com",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/abhishekagarwal2301/isl",
    license="LICENSE",
    description="A package for implementing the Incremental \
        Structure Learning (ISL) algorithm for approximate \
            quantum compiling",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "qiskit~=1.3.1",
        "qiskit_aer~=0.16.0",
        "qiskit_experiments~=0.6.1",
        "numpy",
        "scipy",
        "openfermion~=1.6",
        "sympy",
        "aqc_research @ git+ssh://git@github.com/bjader/aqc-research-qduc.git",
        "physics-tenpy~=1.0.2",
    ],
)
