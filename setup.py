from setuptools import setup, find_packages

setup(
    name="infinite-bookshelf",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'together',
        'python-dotenv',
        'fpdf2',
    ],
)
