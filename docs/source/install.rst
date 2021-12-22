Install mudata
==============

.. contents:: :local:
    :depth: 3

.. toctree::
   :maxdepth: 10

   *

Stable version
--------------

``mudata`` can be installed `from PyPI <https://pypi.org/project/mudata>`_ with ``pip``:
::
	pip install mudata


Development version
-------------------

To use a pre-release version of ``mudata``, install it from `from the GitHub repository <https://github.com/scverse/mudata>`_:
::
	pip install git+https://github.com/scverse/mudata


Troubleshooting
---------------

Please consult the details on installing ``scanpy`` and its dependencies `here <https://scanpy.readthedocs.io/en/stable/installation.html>`_. If there are issues that have not beed described, addressed, or documented, please consider `opening an issue <https://github.com/scverse/mudata/issues>`_.


Hacking on mudata
-----------------
For hacking on the package, it is most convenient to do a so-called development-mode install, which symlinks files in your Python package directory to your mudata working directory, such that you do not need to reinstall after every change. We use `flit <https://flit.readthedocs.io/en/latest/index.html>`_ as our build system. After installing flit, you can run ``flit install -s`` from within the mudata project directory to perform a development-mode install. Happy hacking!
