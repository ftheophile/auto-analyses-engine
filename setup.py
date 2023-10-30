import setuptools

# package can be installed and used just like any other using
# the pip isntall using development mode >> python3 -m pip install -e. (the dot at end is important)
setuptools.setup(
    name="quantlib",
    version="0.1",
    description="Code lib for stock analysis by ftheory",
    url="#",
    author="F. Theory",
    install_requires=["opencv-python"],
    author_email="",
    packages=setuptools.find_packages(),
    zip_safe=False
)