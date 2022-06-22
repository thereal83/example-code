from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

import os

__version__ = "v0.0.1"
__appname__ = "example"


class CacheInstall(install):
    def run(self):
        install.run(self)
        import appdirs
        cache_dir = appdirs.user_data_dir(__appname__)
        os.makedirs(cache_dir, exist_ok=True)

        link = os.path.join(os.path.dirname(__file__), 'cache')
        if not os.path.exists(link):
            os.symlink(
                cache_dir,
                link,
                target_is_directory=True
            )


class CacheDevelop(develop):
    def run(self):
        develop.run(self)
        import appdirs
        cache_dir = appdirs.user_data_dir(__appname__)
        os.makedirs(cache_dir, exist_ok=True)

        link = os.path.join(os.path.dirname(__file__), 'cache')
        if not os.path.exists(link):
            os.symlink(
                cache_dir,
                link,
                target_is_directory=True
            )


setup(
    # Basic info
    name=__appname__,
    version=__version__,
    author="Eric Becker",
    author_email="eric.becker.m@gmail.com",
    description="Code example.",
    python_requires=">=3.7.4",

    # Packages and depencies
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "appdirs==1.4.4",
        "diskcache==4.1.0",
        "numpy==1.22.0",
        "pandas==1.1.0",
        "xgboost==1.1.1",
        "sklearn",
    ],
    extras_require={
        "dev": [
            "ipython",
            "matplotlib",
        ],
    },

    # Post-install script to setup data cache
    cmdclass={
        "install": CacheInstall,
        "develop": CacheDevelop,
    }
)
