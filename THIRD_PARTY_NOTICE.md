# Third-party notice

SNOPT.jl is licensed under the MIT License (see [LICENSE](LICENSE)). It wraps, but
does not distribute, third-party software.

## SNOPT

This package calls into the SNOPT solver library (`libsnopt7`), which is
**closed-source commercial software** owned by Stanford University and the
University of California, San Diego, and distributed by Stanford Business
Software, Inc.

SNOPT is **not** bundled with this package. No SNOPT source code, object code, or
binary is contained in or distributed by this repository. Users must obtain their
own SNOPT license and build or install the shared library themselves; see
<https://ccom.ucsd.edu/~optimizers/solvers/snopt/>.

Use of SNOPT is governed solely by the license agreement between the user and the
SNOPT distributor. The MIT license of SNOPT.jl applies only to the Julia wrapper
code in this repository, and grants no rights to SNOPT itself.

## References

The algorithms implemented by SNOPT are described in:

- P. E. Gill, W. Murray, and M. A. Saunders, "SNOPT: An SQP Algorithm for
  Large-Scale Constrained Optimization", *SIAM Review* 47(1), 2005, pp. 99–131.
