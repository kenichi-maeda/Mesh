# PyMesh Self Intersection Repair

## Current Issue

PyMesh provides `pymesh.resolve_self_intersection()`. <br>
If a mesh is simple, this function can resolve all self intersections. However, if it is complicated, it leaves some intersections.

## How to resolve?
`pymesh.compute_outerhull()` automatically resolves all self intersections.

### 1. Method1
Alternatively, we can simplify vertices of the input mesh by rounding them, along with repeated calls to `pymesh.resolve_self_intersection()`.
Reference: `resolve_intersection.py`

### 2. Method2
Remesh only the intersecting area.
Reference: `fix_mesh.py`

It might be possible to remove all intersecting faces and patch the resulting holes later. However, this approach entails the loss of the original mesh information, as well as the need to fill the holes...