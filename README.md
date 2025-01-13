# PyMesh Self Intersection Repair

## Current Issue

PyMesh provides `pymesh.resolve_self_intersection()`. <br>
If a mesh is simple, this function can resolve all self intersections. However, if it is complicated, it leaves some intersections.

## How to resolve?
### 1. Method1
In the script examples, there is a file called `resolve_intersection.py`. It includes functionality to simplify vertices of the input mesh by rounding them, along with repeated calls to `pymesh.resolve_self_intersection()` to address self-intersections.

### 2. Method2
There is another file called `fix_mesh.py` that globally remeshes a given mesh. The goal of Method2 is to remesh only the intersecting area.

It might be possible to remove all intersecting faces and patch the resulting holes later. However, this approach entails the loss of the original mesh information, as well as the need to fill the holes...