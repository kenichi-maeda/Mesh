# PyMesh Self Intersection Repair

## How to resolve?
`pymesh.resolve_self_intersection()` can fix the issue.
`pymesh.compute_outerhull()` also automatically resolves all self intersections.

### 1. Method1
Alternatively, we can simplify vertices of the input mesh by rounding them, along with repeated calls to `pymesh.resolve_self_intersection()`.
Reference: `resolve_intersection.py`

### 2. Method2
Remesh only the intersecting area.
Reference: `fix_mesh.py`

It might be possible to remove all intersecting faces and patch the resulting holes later. However, this approach entails the loss of the original mesh information, as well as the need to fill the holes...