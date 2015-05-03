from distutils.core import setup

setup(
    name='connorav',
    version='0.1',
    author='C Bates',
    author_email='chrsbats@gmail.com',
    packages=['connorav'],
    scripts=[],
    url='https://github.com/chrsbats/connorav',
    license='LICENSE.TXT',
    description='Fast Generation of Correlated Non-Normal Random Variates',
    long_description='Fast Generation of Correlated Non-Normal Random Variates.  Define your distributions by mean, std, skew and kurtosis and a correlation matrix. Useful for Monte-Carlo testing',
    install_requires=[
        "numpy>=1.9.0", "scipy>=0.14.0"
    ],
)
