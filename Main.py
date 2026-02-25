import py3Dmol
from rdkit import Chem

sdf_path = "100_time_steps_generated_samples.sdf"  # path to your SDF file

# Read SDF
suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)  # keep hydrogens


# Convert to a regular list and filter out invalid molecules
mols = [mol for mol in suppl if mol is not None]

print(f"Total molecules in file: {len(mols)}")

# Take the first molecule
m0 = mols[0]
print(m0)

print("Atoms in the first molecule:")
for atom in m0.GetAtoms():
    idx = atom.GetIdx()
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    print(f"  Index: {idx:2d}, element: {symbol}, charge: {charge}")

num_conformers = m0.GetNumConformers()
print(f"Number of conformers (coordinate sets): {num_conformers}")

if num_conformers > 0:
    conf = m0.GetConformer()
    print("First few atom coordinates:")
    for i in range(min(10, m0.GetNumAtoms())):
        pos = conf.GetAtomPosition(i)
        print(f"  Atom {i:2d}: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")
else:
    print("The molecule has no 3D coordinates (no conformers).")

# 2D image of the molecule
from rdkit.Chem import Draw

img = Draw.MolToImage(m0, size=(300, 300))

# Show using the system image viewer
img.show()

# Or save to a file
img.save("molecule_0.png")
print("Saved image to molecule_0.png")

# Convert the molecule to a MolBlock string (for 3D visualization)
mb = Chem.MolToMolBlock(m0)

view = py3Dmol.view(width=400, height=400)
view.addModel(mb, "sdf")
view.setStyle({'stick': {}})  # you can also use 'sphere', etc.
view.zoomTo()
view.show()
