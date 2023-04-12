from setuptools import setup

exec(open("bolift/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bolift",
    version=__version__,
    description="BayesOPT with LIFT",
    author="Andrew White",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/whitead/bolift",
    license="MIT",
    packages=["bolift"],
    install_requires=[
        "numpy",
        "langchain",
        "openai",
        "faiss-cpu",
        "scipy",
        "pandas",
        "tiktoken",
    ],
    extras_require={"gpr": ["scikit-learn", "torch", "botorch", "gpytorch"]},
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
