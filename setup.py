from setuptools import find_packages, setup

setup(
    name="adaptaqc",
    version="1.0.0",
    author="Ben Jaderberg, George Pennington, Abhishek Agarwal, Kate Marshall, Lewis Anderson",
    author_email="benjamin.jaderberg@ibm.com, george.penngton@stfc.com, kate.marshall@ibm.com, lewis.anderson@ibm.com",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/todo/",
    license="LICENSE",
    description="A package for implementing the Adaptive \
        Approximate Quantum Compiling (ADAPT-AQC) algorithm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "qiskit~=1.3.1",
        "qiskit_aer~=0.16.0",
        "qiskit_experiments~=0.6.1",
        "numpy",
        "scipy",
        "scipy",
        "openfermion~=1.6",
        "sympy",
        "aqc_research @ git+ssh://git@github.com/bjader/aqc-research.git",
        "physics-tenpy~=1.0.2",
    ],
)
