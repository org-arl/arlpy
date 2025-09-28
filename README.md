# ARL Python Tools

[![CI](https://github.com/org-arl/arlpy/workflows/CI/badge.svg)](https://github.com/org-arl/arlpy/actions)

> [!NOTE]
> **This package is now in maintainence mode** and not being actively developed. I personally no longer use
> Python much, as all my scientific work has now moved to Julia. I still maintain this package for others
> who use it, but new features rarely get implemented. I am, however, happy to review and take in PRs from
> contributors who would like to add features or fix bugs.

> [!TIP]
> For those of you who use this package as a Python front-end to BELLHOP, there is now a better option to consider.
> The [`UnderwaterAcoustics.jl`](https://github.com/org-arl/UnderwaterAcoustics.jl) ecosystem provides
> a unified and more capable interface to a variety of propagation models including BELLHOP, KRAKEN, ORCA, etc. Although developed
> for Julia, the packages can [easily be used from Python](https://org-arl.github.io/UnderwaterAcoustics.jl/python.html).

## Introduction

Packages such as `numpy` and `scipy` provide excellent mathematical tools for
scientists and engineers using Python. However, these packages are still young
and evolving, and understandably have some gaps, especially when it comes to
domain-specific requirements. The `arlpy` package aims to fill in some of the
gaps in the areas of underwater acoustics, signal processing, and communication.
Additionally, `arlpy` also includes some commonly needed utilities and plotting
routines based on `bokeh`.

## General modules

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

## Special-purpose modules

The following modules are specific to tools available at the ARL and may not be of
general interest to others:

* Digital Towed Array (`arlpy.dtla`)
* ROMANIS (`arlpy.romanis`)
* HiDAQ (`arlpy.hidaq`)
* UNET (`arlpy.unet`)

## Usage

Installation::
```
pip install arlpy
```

To import all general modules::
```
import arlpy
```

## Notes

Png export of bokeh plots requires `selenium`, `pillow` and `phantomjs`. These are not
installed as automatic depdendencies, since they are optional and only required
for png export. These should be installed manually, if desired.

## Citing

```
@software{arlpy,
  author = {Mandar Chitre},
  title = {{arlpy}: ARL Python Tools},
  version = {1.9.1},
  year = {2024},
  url = {https://github.com/org-arl/arlpy/tree/v1.9.1}
}
```

## Useful links

* [arlpy home](https://github.com/org-arl/arlpy)
* [arlpy on PyPi](https://pypi.org/project/arlpy/)
* [arlpy documentation](https://arlpy.readthedocs.io/en/latest/)
