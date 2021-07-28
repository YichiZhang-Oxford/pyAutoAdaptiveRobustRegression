from setuptools import setup
setup(
  name = 'pyAutoAdaptiveRobustRegression',         # How you named your package folder (MyLib)
  packages = ['pyAutoAdaptiveRobustRegression'],   # Chose the same as "name"
  version = '0.1.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Auto Adaptive Robust Regression Python Package',   # Give a short description about your library
  author = 'Yichi Zhang',                   # Type in your name
  author_email = 'yichi.zhang@worc.ox.ac.uk',      # Type in your E-Mail
  url = 'https://github.com/YichiZhang-Oxford/pyAutoAdaptiveRobustRegression',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/YichiZhang-Oxford/pyAutoAdaptiveRobustRegression',    # I explain this later on
  keywords = ['Auto Adaptive Robust Regression'],   # Keywords that define your package best
  package_data={
    'pyAutoAdaptiveRobustRegression': [
        'bin/linux/pyAutoAdaptiveRobustRegression.so',
        'bin/macos/pyAutoAdaptiveRobustRegression.dylib',
        'bin/win32/pyAutoAdaptiveRobustRegression.dll',
    ]
  },
  install_requires = ['numpy'],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
