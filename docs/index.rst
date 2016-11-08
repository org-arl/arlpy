ARL Python Tools
================

Packages such as `numpy` and `scipy` provide excellent mathematical tools for
scientists and engineers using Python. However, these packages are still young
and evolving, and understandably have some gaps, especially when it comes to
domain-specific requirements. The `arlpy` package aims to fill in some of the
gaps in the areas of underwater acoustics, signal processing, and communication.

General modules
---------------

The following modules are general and are likely to be of interest to researchers
and developers working on signal processing, communication and underwater acoustics:

.. toctree::
   :maxdepth: 1

   signal
   comms
   geo
   uwa
   utils

Special-purpose modules
-----------------------

The following modules are specific to tools available at the ARL and may not be of
general interest to others:

.. toctree::
   :maxdepth: 1

   dtla
   hidaq
