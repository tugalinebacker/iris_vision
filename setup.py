from distutils.core import setup
 
setup(
    version='0.0.0',
    scripts=['src/inference_mkII.py', 'src/Doppler.py'],
    packages=['iris_vision','object_detection'],
    package_dir={'': 'src'}
)
