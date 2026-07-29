#!/usr/bin/env python
"""Render generated-lysozyme ESMFold structures as pLDDT-coloured cartoons.

Run with the PyMOL-open-source environment:
    ~/miniconda3/envs/pymolviz/bin/pymol -cq r2_interpretability_transfer/scripts/render_lysozyme_cartoons.py

Outputs ray-traced PNGs (transparent background) used by Figure 5 of the
R2 manuscript. pLDDT is stored in the B-factor column on a 0-1 scale and
coloured with the AlphaFold/ESMFold confidence palette.
"""

from pymol import cmd

LEADS = ["lead_0004", "lead_0009", "lead_0008"]
BASE = "r2_interpretability_transfer/results/drug_design/ec_lysozyme_esmfold_metrics_pdbs"
OUT = "r2_interpretability_transfer/results/drug_design/pymol_renders"

cmd.set_color("plddt_vhigh", [0 / 255, 83 / 255, 214 / 255])    # >0.90
cmd.set_color("plddt_high", [101 / 255, 203 / 255, 243 / 255])  # 0.70-0.90
cmd.set_color("plddt_low", [255 / 255, 219 / 255, 19 / 255])    # 0.50-0.70
cmd.set_color("plddt_vlow", [255 / 255, 125 / 255, 69 / 255])   # <0.50

cmd.bg_color("white")
cmd.set("ray_opaque_background", 0)
cmd.set("ray_shadows", 0)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_smooth_loops", 1)
cmd.set("antialias", 2)
cmd.set("ambient", 0.38)
cmd.set("specular", 0.15)

for name in LEADS:
    cmd.delete("all")
    cmd.load(f"{BASE}/{name}.pdb", name)
    cmd.hide("everything")
    cmd.show("cartoon")
    # cumulative high->low coverage so every residue is coloured (no gaps)
    cmd.color("plddt_vhigh", name)
    cmd.color("plddt_high", f"{name} and b<0.9")
    cmd.color("plddt_low", f"{name} and b<0.7")
    cmd.color("plddt_vlow", f"{name} and b<0.5")
    cmd.orient(name)
    cmd.zoom(name, buffer=3, complete=1)
    cmd.ray(1200, 1200)
    cmd.png(f"{OUT}/{name}.png", dpi=300)
    print("rendered", name)
