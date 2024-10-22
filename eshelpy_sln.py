"""
Ahora que tienes la extensión de Python y el entorno virtual configurado, selecciona el entorno adecuado:

Abre la paleta de comandos (Ctrl+Shift+P).
Escribe Python: Select Interpreter y selecciona esta opción.
En la lista de entornos, selecciona el entorno virtual o la instalación específica de Python que desees usar.

"""
#%%
from mpi4py import MPI
import dolfinx # FEM in python
import matplotlib.pyplot as plt
import ufl # variational formulations
import numpy as np
import gmsh
import basix.ufl
import dolfinx.fem as fem
import dolfinx.fem.petsc
#%%
import pyvista # visualisation in python notebook
pyvista.start_xvfb()
pyvista.set_jupyter_backend("static")
# %%
import utils

#%%
# Geometry
R_i = 1.0 # Radius of the inclusion
R_e = 6.9  # Radius of the matrix (whole domain)
aspect_ratio = 1.0 # start with a circle, otherwise ellipse

# Material
E_m = 1.0 # Young's modulus in matrix
nu_m = 0.35 # Poisson's ratio in matrix
E_i = 11.0 # Young's modulus of inclusion
nu_i = 0.3 # Poisson's ratio in inclusion

## Create the mesh with gmsh

R_i = 1.0 # Radius of the inclusion
R_e = 6.9  # Radius of the matrix (whole domain)
aspect_ratio = 1.0
mesh_size = 0.2*R_i
mesh_order = 1

mesh_comm = MPI.COMM_WORLD
model_rank = 0
gmsh.initialize()
facet_names = {"inner_boundary": 1, "outer_boundary": 2}
cell_names = {"inclusion": 1, "matrix": 2}
model = gmsh.model()
model.add("Disk")
model.setCurrent("Disk")
gdim = 2 # geometric dimension of the mesh
inner_disk = gmsh.model.occ.addDisk(0, 0, 0, R_i, aspect_ratio * R_i)
outer_disk = gmsh.model.occ.addDisk(0, 0, 0, R_e, R_e)
whole_domain = gmsh.model.occ.fragment(
            [(gdim, outer_disk)], [(gdim, inner_disk)]
        )
gmsh.model.occ.synchronize()
# Add physical tag for bulk
inner_domain = whole_domain[0][0]
outer_domain = whole_domain[0][1]
model.addPhysicalGroup(gdim, [inner_domain[1]], tag=cell_names["inclusion"])
model.setPhysicalName(gdim, inner_domain[1], "Inclusion")
model.addPhysicalGroup(gdim, [outer_domain[1]], tag=cell_names["matrix"])
model.setPhysicalName(gdim, outer_domain[1], "Matrix")

# Add physical tag for boundaries
lines = gmsh.model.getEntities(dim=1)
inner_boundary = lines[1][1]
outer_boundary = lines[0][1]
gmsh.model.addPhysicalGroup(1, [inner_boundary], facet_names["inner_boundary"])
gmsh.model.addPhysicalGroup(1, [outer_boundary], facet_names["outer_boundary"])
gmsh.option.setNumber("Mesh.CharacteristicLengthMin",mesh_size)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",mesh_size)
model.mesh.generate(gdim)
gmsh.option.setNumber("General.Terminal", 1)
model.mesh.setOrder(mesh_order)
gmsh.option.setNumber("General.Terminal", 0)

# Import the mesh in dolfinx
from dolfinx.io import gmshio
domain, cell_tags, facet_tags = gmshio.model_to_mesh(model, mesh_comm, model_rank, gdim=gdim)
domain.name = "composite"
cell_tags.name = f"{domain.name}_cells"
facet_tags.name = f"{domain.name}_facets"
gmsh.finalize()

cell_names["matrix"]

cell_names["inclusion"]

# `Questions start here`

# 0) Export the mesh as a xdmf file and open it in Paraview

# Save the mesh in XDMF format
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/mesh.xdmf", "w") as file:
    file.write_mesh(domain)
    domain.topology.create_connectivity(1, 2)

# 1) Plot the mesh with a color code to locate the inclusion and the matrix

topology, cells, geometry = dolfinx.plot.vtk_mesh(domain)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)
function_grid["Marker"] = cell_tags.values
plotter = pyvista.Plotter()
plotter.add_mesh(function_grid, show_edges=True)
plotter.show_bounds(grid='front', location='outer', all_edges=True)
plotter.view_xy()
plotter.show()
# %%
