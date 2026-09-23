DESRES Data Sets (DESS66x8)
=========================================
Please see the original paper at https://doi.org/10.1038/s41597-021-00833-x for
more information about this dataset.

This package contains a datasets described by Donchev et al. [1]: DESS66x8, It
is presented as a CSV (DESS66x8.csv) and .mol files
(geometries/<system_id>/DESS66x8_<geom_id>.mol). Also included is a metadata
file DESS66x8_meta.csv, which contains a set of long-form column descriptions
replicating those in [1], as well as data types and units (when applicable) for
each column.

Manifest
--------
- DESS66x8.csv       : S66x8 geometries computed in the same style as DES370K,
                       containing interaction energies calculated using
                       CCSD(T), MP2, HF, and SAPT0, as well as dimer
                       geometries, and bronze-standard [3] reference values.

- DESS66x8_meta.csv  : Long-form descriptions of the columns in DESS66x8, as
                       well as datatypes and units (when applicable) for each
                       column

- LICENSE.txt        : License for using and redistributing the datasets
                       provided.

- README.md          : This file.

Loading the Datset
------------------
The datasets are presented as CSVs as a compromise between human-readability,
format uniformity, and parsing speed. While an almost uncountable number of
packages exist to read CSV files, we recommend using the python data analysis

References
----------
[1]  A. G. Donchev, A. G. Taube, E. Decolvenaere, C. Hargus, R. T. McGibbon,
      K.-H. Law, B. A. Gregersen, J.-L. Li, K. Palmo, K. Siva, M. Bergdorf, 
      J. L. Klepeis, and D. E. Shaw. "Quantum chemical benchmark database of
      dimer interaction energies at a “gold standard” level of accuracy" 

[2]  R. T. McGibbon, A. G. Taube, A. G. Donchev, K. Siva, F. Fernandez, C. Hargus,
      K.-H. Law, J.L. Klepeis, and D. E. Shaw. "Improving the accuracy of
      Moller-Plesset perturbation theory with neural networks"

[3]  M. K. Kesharwani, A. Karton, N. Sylvetsky, J. M. L. Nitai. "The S66
      non-covalent interactions benchmark reconsidered using explicitly
      correlated methods near the basis set limit."

License
-------
```
                DESRES DATA SETS LICENSE AGREEMENT

Copyright 2020, D. E. Shaw Research. All rights reserved.

Redistribution and use of electronic structure data released in the DESRES
Data Sets (DES370K, DES15K, DES5M, DESS66, and DESS66x8) with or without
modification, is permitted provided that the following conditions are met:

    * Redistributions of the data must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions, and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

Neither the name of D. E. Shaw Research nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE AND DATA ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDINGNEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE AND/OR DATA, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
