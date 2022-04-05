from setuptools import setup, find_packages

setup(name='regionsLoader', 
    version='0.1', 
    packages=find_packages(),
    include_package_data=True,
    install_requires=['cooler', 'pybbi', 'keras', 'bioframe'],

    )