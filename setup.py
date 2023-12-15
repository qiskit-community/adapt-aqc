from setuptools import find_packages, setup

setup(
    name="quantum-isl",
    version="1.0.1",
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
        "qiskit~=0.45.1",
        "qiskit_aer~=0.13.1",
        "qiskit_experiments~=0.5.4",
        "qiskit_ibmq_provider~=0.20.2",
        "qiskit_terra~=0.45.1",
        "numpy",
        "scipy",
        "openfermion~=1.6",
        "sympy"
    ],
)
