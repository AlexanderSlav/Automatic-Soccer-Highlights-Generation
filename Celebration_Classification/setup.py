import codecs
import os.path

from setuptools import setup


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_version(rel_path):
    with codecs.open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), rel_path), "r",
    ) as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().split("\n")

packages = ["soccer_summarizator"]

package_data = {
    "soccer_summarizator": ["checkpoints/*"],
}

setup_kwargs = {
    "name": "soccer_summarizator",
    "version": get_version("soccer_summarizator/__init__.py"),
    "description": "Soccer Goals Summarizator based on detecting Celebration Events and make offset n frames back",
    "long_description": get_long_description(),
    "author": "Slavutin Alexander",
    "author_email": "alexander.slavutin@gmail.com",
    "url": "https://github.com/AlexanderSlav/Automatic-Soccer-Highlights-Generation/tree/main/Celebration_Classification",
    "license": "MIT",
    "packages": packages,
    "package_data": package_data,
    "python_requires": ">=3.7,<4.0",
    "install_requires": requirements,
    "long_description_content_type": "text/markdown"
}

get_long_description()
setup(**setup_kwargs)
