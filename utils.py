import pymesh
import numpy as np
import pyvista as pv
from IPython.display import display
import logging
from tabulate import tabulate
import os
from scipy.spatial import KDTree

def process_mesh(mesh, mesh_processing_function):
    vertices_before = mesh.vertices
    faces_before = mesh.faces
    intersections_before = pymesh.detect_self_intersection(mesh)
    mesh = mesh_processing_function(mesh)
    vertices_after = mesh.vertices
    faces_after = mesh.faces
    intersections_after = pymesh.detect_self_intersection(mesh)

    table_data = [
        ["Metric", "Before", "After"],
        ["Vertices", len(vertices_before), len(vertices_after)],
        ["Faces", len(faces_before), len(faces_after)],
        ["Intersecting Pairs", len(intersections_before), len(intersections_after)],
    ]
    
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    return mesh

def remove_duplicated_vertices(mesh):
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    return mesh

def remove_duplicated_faces(mesh):
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    return mesh

def remove_isolated_vertices(mesh):
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    return mesh

def remove_obtuse_triangles(mesh):
    mesh, _ = pymesh.remove_obtuse_triangles(mesh)
    return mesh

def remove_degenerated_triangles(mesh):
    mesh, _ = pymesh.remove_degenerated_triangles(mesh)
    return mesh

def resolve_self_intersection(mesh):
    mesh = pymesh.resolve_self_intersection(mesh)
    return mesh

def self_intersection_stats(mesh):
    intersections = pymesh.detect_self_intersection(mesh)
    table_data = [
        ["Metric", "Value"],
        ["Number of vertices", len(mesh.vertices)],
        ["Number of faces", len(mesh.faces)],
        ["Number of intersecting face pairs", len(intersections)],
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))


def convert_to_pyvista(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    faces_flat = np.hstack([[3, *face] for face in faces]).astype(np.int32)
    return pv.PolyData(vertices, faces_flat)


def visualize(mesh, filename="Processed Mesh"):

    mesh = convert_to_pyvista(mesh)
    plotter = pv.Plotter(notebook=True)

    # Add the mesh
    mesh_actor = plotter.add_mesh(
        mesh,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename,
        line_width=0.3
    )

    def update_clipping_plane(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh.clip_surface(new_plane, invert=False)
        mesh_actor.mapper.SetInputData(clipped_mesh)
        plotter.render()

    plotter.add_slider_widget(
        callback=update_clipping_plane,
        rng=[-2, 2],
        value=0,
        title="Clip Plane",
    )

    plotter.add_legend()
    display(plotter.show(jupyter_backend="trame"))

def intermediate(mesh):
    temp_file = "data/temp.ply"
    pymesh.save_mesh(temp_file, mesh, ascii=True)
    mesh = pymesh.load_mesh(temp_file)
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return mesh

def extract_submesh_by_bbox(mesh, local_min, local_max):
    """
    Extracts a submesh from the input mesh that includes only the faces whose
    vertices lie within the specified axis-aligned bounding box [local_min, local_max].

    Parameters:
        mesh: The input mesh object.
        local_min: A 1D array defining the minimum (x, y, z) coordinates of the bounding box.
        local_max: A 1D array defining the maximum (x, y, z) coordinates of the bounding box.

    Returns:
        sub_mesh: The extracted submesh.
        face_mask: A boolean array indicating which faces of the original mesh were included in the submesh.
    """
    verts = mesh.vertices
    # inside_mask[i] is True if all x, y, and z in verts[i] lie within [local_min, local_max].
    inside_mask = np.all((verts >= local_min) & (verts <= local_max), axis=1)

    faces = mesh.faces
    # face_mask[i] is True if at least one of the three vertices of faces[i] lies within [local_min, local_max]
    face_mask = inside_mask[faces].any(axis=1)

    # Extract the subset of faces that satisfy the condition.
    sub_faces = faces[face_mask]

    # Create a submesh.
    sub_mesh = pymesh.form_mesh(verts, sub_faces)

    # Remove isolated vertices to ensure a clean submesh.
    sub_mesh, _ = pymesh.remove_isolated_vertices(sub_mesh)
    
    return sub_mesh, face_mask

def extract_remaining_mesh(original_mesh, face_mask):
    """
    Remove the faces of the submesh from the original mesh and return the remaining part.
    """
    faces = original_mesh.faces
    keep_mask = np.logical_not(face_mask)
    kept_faces = faces[keep_mask]
    remaining_mesh = pymesh.form_mesh(original_mesh.vertices, kept_faces)
    remaining_mesh, _ = pymesh.remove_isolated_vertices(remaining_mesh)
    return remaining_mesh

def detect_boundary_vertices(mesh):
    """
    Detect boundary vertices in the given mesh.
    A boundary vertex is connected to at least one boundary edge.
    """
    from collections import defaultdict

    # Step 1: Build an edge-to-face mapping
    edge_to_faces = defaultdict(list)
    for face_idx, face in enumerate(mesh.faces):
        edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)

    # Step 2: Identify boundary edges (edges with only one adjacent face)
    boundary_edges = [edge for edge, faces in edge_to_faces.items() if len(faces) == 1]

    # Step 3: Collect boundary vertices
    boundary_vertices = set()
    for edge in boundary_edges:
        boundary_vertices.update(edge)

    # Step 4: Create a mask for boundary vertices
    boundary_mask = np.zeros(mesh.num_vertices, dtype=bool)
    boundary_mask[list(boundary_vertices)] = True

    return boundary_mask


def align_submesh_boundary(remaining_mesh, repaired_submesh, tolerance=4):
    """
    Align the boundary vertices of the repaired submesh to the original mesh's boundary.
    """

    # Identify boundary vertices
    original_boundary = detect_boundary_vertices(remaining_mesh)
    repaired_boundary = detect_boundary_vertices(repaired_submesh)

    # Build KDTree for original mesh boundary vertices
    original_vertices = remaining_mesh.vertices
    repaired_vertices = repaired_submesh.vertices
    tree = KDTree(original_vertices[original_boundary])

    repaired_vertices = repaired_submesh.vertices.copy()
    # Snap repaired boundary vertices to original mesh
    for idx in np.where(repaired_boundary)[0]:  # Iterating over the indices of vertices marked as boundary vertices
        dist, nearest_idx = tree.query(repaired_vertices[idx], distance_upper_bound=tolerance)
        if dist < tolerance:
            # For example, suppose tolerance = 0.2
            # [1.1, 0, 0] becomes [1.0, 0, 0]
            repaired_vertices[idx] = original_vertices[original_boundary][nearest_idx]

    # Rebuild submesh with aligned vertices
    repaired_submesh = pymesh.form_mesh(repaired_vertices, repaired_submesh.faces)
    return repaired_submesh


def replace_submesh_in_original(remaining_mesh, repaired_submesh):
    """
    Re-stitche a repaired submesh back into the original mesh.

    Steps:
    1. Remove the old faces from the original (the region we're replacing).
    2. Combine the leftover portion of the original with the newly repaired submesh
       using boolean union or just form_mesh + remove_duplicates.
    """
    merged = pymesh.merge_meshes([remaining_mesh, repaired_submesh])
    merged, _ = pymesh.remove_duplicated_faces(merged)
    merged, _ = pymesh.remove_isolated_vertices(merged)
    return merged

