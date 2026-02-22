Installation
============

Requirements
------------

* Python >= 3.10
* numpy, scipy, matplotlib, pandas, lmfit (installed automatically)

From source (development install)
----------------------------------

ScatterBrain uses `uv <https://docs.astral.sh/uv/>`_ for dependency management.

.. code-block:: bash

   git clone https://github.com/your_username/ScatterBrain.git
   cd ScatterBrain
   uv sync --all-extras

Run commands inside the managed environment with the ``uv run`` prefix:

.. code-block:: bash

   uv run python -c "import scatterbrain; print(scatterbrain.__version__)"

Alternatively, activate the virtual environment directly:

.. code-block:: bash

   source .venv/bin/activate

From PyPI
---------

Once the package is published:

.. code-block:: bash

   uv add scatterbrain
   # or: pip install scatterbrain
