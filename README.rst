ARL Python Tools
================

Packages such as `numpy` and `scipy` provide excellent mathematical tools for
scientists and engineers using Python. However, these packages are still young
and evolving, and understandably have some gaps, especially when it comes to
domain-specific requirements. The `arlpy` package aims to fill in some of the
gaps in the areas of underwater acoustics, signal processing, and communication.
Additionally, `arlpy` also includes some commonly needed utilities and plotting
routines based on `bokeh`.

General modules
---------------

The following modules are general and are likely to be of interest to researchers
and developers working on signal processing, communication and underwater acoustics:

    * Signal processing (`arlpy.signal`)
    * Communications (`arlpy.comms`)
    * Beamforming and array processing (`arlpy.bf`)
    * Stable distributions (`arlpy.stable`)
    * Geographical coordinates (`arlpy.geo`)
    * Underwater acoustics (`arlpy.uwa`)
    * Underwater acoustic propagation modeling (`arlpy.uwapm`)
    * Plotting utilities (`arlpy.plot`)
    * Common utilities (`arlpy.utils`)

Special-purpose modules
-----------------------

The following modules are specific to tools available at the ARL and may not be of
general interest to others:

    * Digital Towed Array (`arlpy.dtla`)
    * ROMANIS (`arlpy.romanis`)
    * HiDAQ (`arlpy.hidaq`)
    * UNET (`arlpy.unet`)

Usage
-----

Installation::

    pip install arlpy

To import all general modules::

    import arlpy

Notes
-----

Png export of bokeh plots requires `selenium`, `pillow` and `phantomjs`. These are not
installed as automatic depdendencies, since they are optional and only required
for png export. These should be installed manually, if desired.

Useful links
------------

    * `arlpy home <https://github.com/org-arl/arlpy>`_
    * `arlpy documentation <http://arlpy.readthedocs.io>`_
