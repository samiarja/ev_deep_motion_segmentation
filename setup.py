import builtins
import setuptools
import setuptools.extension
import setuptools.command.build_ext
import sys

eigen_path = "./eigen-3.4.0"

with open("README.md") as file:
    long_description = file.read()


class build_ext(setuptools.command.build_ext.build_ext):
    def finalize_options(self):
        setuptools.command.build_ext.build_ext.finalize_options(self)
        builtins.__NUMPY_SETUP__ = False  # type: ignore
        import numpy

        self.include_dirs.append(numpy.get_include())
        self.include_dirs.append(eigen_path)  # Add Eigen path here


extra_args = []
if sys.platform == "linux":
    extra_args += ["-std=c++11"]
elif sys.platform == "darwin":
    extra_args += ["-std=c++11", "-stdlib=libc++"]

setuptools.setup(
    name="ev_deep_motion_segmentation",
    version="0.0.1",
    author="ICNS, Sami Arja",
    author_email="sami.arja@gmail.com",
    description="Event-based motion segmentation for aerial surveillance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=['numpy'],
    install_requires=[
        "cmaes >= 0.8.2",
        "event_stream >= 1.4.1",
        "h5py >= 3.7.0",
        "matplotlib >= 3.5.2",
        "numpy >= 1.23.1",
        "pillow >= 9.2.0",
        "scipy >= 1.8.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=["event_warping"],
    ext_modules=[
        setuptools.extension.Extension(
            "event_warping_extension",
            language="c++",
            sources=["event_warping_extension/event_warping_extension.cpp"],
            include_dirs=[],  # Leave this empty, it will be filled in build_ext
            libraries=[],
            extra_compile_args=extra_args,
            extra_link_args=extra_args,
        ),
    ],
    cmdclass={"build_ext": build_ext},
)

