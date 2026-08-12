load raw/4jwr_protein.pdb, protein
load raw/4jwr_ligand.sdf, ligand
hide all
show surface, protein
color white, protein
set transparency, 0.4, protein
show sticks, ligand
util.cbay ligand
pseudoatom p1_0, pos=[4.179, -16.422, 0.475]
pseudoatom p2_0, pos=[5.121, -14.976, 0.371]
distance dist_0, p1_0, p2_0
color magenta, dist_0
hide labels, dist_0
pseudoatom p1_1, pos=[4.887, -13.946, 1.284]
pseudoatom p2_1, pos=[5.121, -14.976, 0.371]
distance dist_1, p1_1, p2_1
color magenta, dist_1
hide labels, dist_1
pseudoatom p1_2, pos=[6.104, -14.845, -0.615]
pseudoatom p2_2, pos=[6.851, -13.667, -0.675]
distance dist_2, p1_2, p2_2
color magenta, dist_2
hide labels, dist_2
pseudoatom p1_3, pos=[6.104, -14.845, -0.615]
pseudoatom p2_3, pos=[5.121, -14.976, 0.371]
distance dist_3, p1_3, p2_3
color magenta, dist_3
hide labels, dist_3
pseudoatom p1_4, pos=[6.851, -13.667, -0.675]
pseudoatom p2_4, pos=[6.104, -14.845, -0.615]
distance dist_4, p1_4, p2_4
color magenta, dist_4
hide labels, dist_4
pseudoatom p1_5, pos=[9.758, -11.332, -0.728]
pseudoatom p2_5, pos=[8.859, -11.550, 0.357]
distance dist_5, p1_5, p2_5
color magenta, dist_5
hide labels, dist_5
pseudoatom p1_6, pos=[7.404, -11.315, 0.205]
pseudoatom p2_6, pos=[6.609, -12.632, 0.238]
distance dist_6, p1_6, p2_6
color magenta, dist_6
hide labels, dist_6
pseudoatom p1_7, pos=[6.609, -12.632, 0.238]
pseudoatom p2_7, pos=[6.851, -13.667, -0.675]
distance dist_7, p1_7, p2_7
color magenta, dist_7
hide labels, dist_7
pseudoatom p1_8, pos=[5.712, -14.353, 4.945]
pseudoatom p2_8, pos=[5.509, -15.657, 5.163]
distance dist_8, p1_8, p2_8
color magenta, dist_8
hide labels, dist_8
pseudoatom p1_9, pos=[6.344, -10.532, -5.380]
pseudoatom p2_9, pos=[5.468, -11.423, -5.947]
distance dist_9, p1_9, p2_9
color magenta, dist_9
hide labels, dist_9
pseudoatom p1_10, pos=[2.799, -12.305, -3.452]
pseudoatom p2_10, pos=[1.513, -13.118, -7.305]
distance dist_10, p1_10, p2_10
color magenta, dist_10
hide labels, dist_10
pseudoatom p1_11, pos=[9.577, -9.848, -4.846]
pseudoatom p2_11, pos=[7.278, -10.757, -5.066]
distance dist_11, p1_11, p2_11
color magenta, dist_11
hide labels, dist_11
pseudoatom p1_12, pos=[9.210, -9.111, -3.899]
pseudoatom p2_12, pos=[6.200, -8.454, -4.886]
distance dist_12, p1_12, p2_12
color magenta, dist_12
hide labels, dist_12
pseudoatom p1_13, pos=[9.210, -9.111, -3.899]
pseudoatom p2_13, pos=[7.278, -10.757, -5.066]
distance dist_13, p1_13, p2_13
color magenta, dist_13
hide labels, dist_13
pseudoatom p1_14, pos=[5.712, -14.353, 4.945]
pseudoatom p2_14, pos=[5.121, -14.976, 0.371]
distance dist_14, p1_14, p2_14
color magenta, dist_14
hide labels, dist_14
pseudoatom p1_15, pos=[7.407, -17.852, 2.091]
pseudoatom p2_15, pos=[5.121, -14.976, 0.371]
distance dist_15, p1_15, p2_15
color magenta, dist_15
hide labels, dist_15
set dash_gap, 0.1
set dash_radius, 0.08
zoom ligand
