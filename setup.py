import re

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()
    # Remove p tags.
    pattern = re.compile(r"<p.*?>.*?</p>", re.DOTALL)
    long_description = re.sub(pattern, "", long_description)

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [
        line
        for line in f.read().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="datastorm",
    version="0.1.0",
    author=(
        "Shicheng Liu, Yucheng Jiang, Sajid Farook, Camila Nicollier Sanchez, "
        "David Fernando Castro Pena, Monica S. Lam"
    ),
    description=(
        "DataSTORM: Deep Research on Large-Scale Databases using Exploratory "
        "Data Analysis and Data Storytelling"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stanford-oval/datastorm",
    license="Apache License 2.0",
    packages=find_packages(include=["knowledge_storm", "knowledge_storm.*"]),
    package_data={"knowledge_storm": ["datastorm/modules/*.sh"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
