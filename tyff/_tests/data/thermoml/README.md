# ThermoML data

* test.sqlite: an existing CoordinateStore. This contains 21 boxes, where
  * ids 1-4: water (1000 mols)
    * 1: 313.15 K, 101 kPa in atms
    * 2: 298.15 K, 1 atm
    * 3: 298.15 K, no pressure
    * 4: same thermodynamic state as 1, atoms are rearranged
  * ids 5-8: isopropanol (1000 mols), with the same breakdown as water above
  * ids 9-12: water/isopropanol mixture (1000 mols), same breakdown as above
  * ids 13-15: water/triethanolamine mixture (2000 mols):
    * 13: 298.15 K, 101 kPa in atms
    * 14: 313.15 K, 101 kPa in atms, with atoms rearranged
    * 15: 298.15 K, no pressure
  * ids 16-18: water (2000 mols), same breakdown as triethanolamine mixture above
  * ids 19-21: triethanolamine (2000 mols), same breakdown as triethanolamine mixture above
