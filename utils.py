import pymesh
import numpy as np
import pyvista as pv
from IPython.display import display
import logging
from tabulate import tabulate
import os
from scipy.spatial import KDTree
import trimesh
from scipy.spatial import cKDTree
from numpy.linalg import norm
import pymeshfix

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

def visualize_two_meshes(mesh1, mesh2, filename1="Mesh 1", filename2="Mesh 2"):
    mesh1 = convert_to_pyvista(mesh1)
    mesh2 = convert_to_pyvista(mesh2)
    plotter = pv.Plotter(shape=(1, 2), notebook=True)

    plotter.subplot(0, 0)
    mesh_actor1 = plotter.add_mesh(
        mesh1,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename1,
        line_width=0.3
    )
    plotter.add_text("Mesh 1", position="upper_left", font_size=10)

    def update_clipping_plane_mesh1(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh1.clip_surface(new_plane, invert=False)
        mesh_actor1.mapper.SetInputData(clipped_mesh)
        plotter.render()

    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh1,
        rng=[-2, 2],
        value=0,
        title="Clip Plane Mesh 1",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )

    plotter.subplot(0, 1)
    mesh_actor2 = plotter.add_mesh(
        mesh2,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename2,
        line_width=0.3
    )
    plotter.add_text("Mesh 2", position="upper_left", font_size=10)

    def update_clipping_plane_mesh2(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh2.clip_surface(new_plane, invert=False)
        mesh_actor2.mapper.SetInputData(clipped_mesh)
        plotter.render()

    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh2,
        rng=[-2, 2],
        value=0,
        title="Clip Plane Mesh 2",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )

    plotter.add_legend()
    display(plotter.show(jupyter_backend="trame"))

def visualize_three_meshes(mesh1, mesh2, mesh3, filename1="Mesh 1", filename2="Mesh 2", filename3="Mesh 3"):
    mesh1 = convert_to_pyvista(mesh1)
    mesh2 = convert_to_pyvista(mesh2)
    mesh3 = convert_to_pyvista(mesh3)
    
    plotter = pv.Plotter(shape=(1, 3), notebook=True)

    # Mesh 1
    plotter.subplot(0, 0)
    mesh_actor1 = plotter.add_mesh(
        mesh1,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename1,
        line_width=0.3
    )
    plotter.add_text("Mesh 1", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh1(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh1.clip_surface(new_plane, invert=False)
        mesh_actor1.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh1,
        rng=[-2, 2],
        value=0,
        title="Clip Plane Mesh 1",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )

    # Mesh 2
    plotter.subplot(0, 1)
    mesh_actor2 = plotter.add_mesh(
        mesh2,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename2,
        line_width=0.3
    )
    plotter.add_text("Mesh 2", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh2(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh2.clip_surface(new_plane, invert=False)
        mesh_actor2.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh2,
        rng=[-2, 2],
        value=0,
        title="Clip Plane Mesh 2",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )
    
    # Mesh 3
    plotter.subplot(0, 2)
    mesh_actor3 = plotter.add_mesh(
        mesh3,
        color="white",
        show_edges=True,
        edge_color="black",
        label=filename3,
        line_width=0.3
    )
    plotter.add_text("Mesh 3", position="upper_left", font_size=10)
    
    def update_clipping_plane_mesh3(value):
        new_plane = pv.Plane(center=(value, 0, 0), direction=(1, 0, 0))
        clipped_mesh = mesh3.clip_surface(new_plane, invert=False)
        mesh_actor3.mapper.SetInputData(clipped_mesh)
        plotter.render()
    
    plotter.add_slider_widget(
        callback=update_clipping_plane_mesh3,
        rng=[-2, 2],
        value=0,
        title="Clip Plane Mesh 3",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9)
    )
    
    plotter.add_legend()
    display(plotter.show(jupyter_backend="trame"))


def visualize_intersection(mesh, filename="Processed Mesh"):
    intersections = pymesh.detect_self_intersection(mesh)
    intersecting_faces = set(intersections.flatten())

    mesh = convert_to_pyvista(mesh)

    scalars = [
        1 if i in intersecting_faces else 0
        for i in range(mesh.n_faces)
    ]
    mesh.cell_data["intersections"] = scalars

    plotter = pv.Plotter(notebook=True)

    mesh_actor = plotter.add_mesh(
        mesh,
        scalars="intersections",
        color="white",
        show_edges=True,
        edge_color="black",
        line_width=0.3,
        cmap=["white", "red"],  # White for normal, red for intersecting
        label=filename,
        show_scalar_bar=False
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

def evaluation(mesh1, mesh2):
    before_trimesh = trimesh.Trimesh(vertices=mesh1.vertices, faces=mesh1.faces)
    after_trimesh = trimesh.Trimesh(vertices=mesh2.vertices, faces=mesh2.faces)

    before_intersections = pymesh.detect_self_intersection(mesh1)
    after_intersections = pymesh.detect_self_intersection(mesh2)

    table_data = [
        ["Metric", "Before", "After"],
        ["Number of vertices", len(mesh1.vertices), len(mesh2.vertices)],
        ["Number of faces", len(mesh1.faces), len(mesh2.faces)],
        ["Number of intersecting face pairs", len(before_intersections), len(after_intersections)],
        ["Volume", before_trimesh.volume, after_trimesh.volume],
        ["Area", before_trimesh.area, after_trimesh.area]
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

def full_evaluation(original, mesh1, mesh2, mesh3):
    before_trimesh = trimesh.Trimesh(vertices=original.vertices, faces=original.faces)
    after_trimesh1 = trimesh.Trimesh(vertices=mesh1.vertices, faces=mesh1.faces)
    after_trimesh2 = trimesh.Trimesh(vertices=mesh2.vertices, faces=mesh2.faces)
    after_trimesh3 = trimesh.Trimesh(vertices=mesh3.vertices, faces=mesh3.faces)

    before_intersections = pymesh.detect_self_intersection(original)
    after_intersections1 = pymesh.detect_self_intersection(mesh1)
    after_intersections2 = pymesh.detect_self_intersection(mesh2)
    after_intersections3 = pymesh.detect_self_intersection(mesh3)

    table_data = [
        ["Metric", "Original", "Method1", "Method2", "Method3"],
        ["Number of vertices", len(original.vertices), len(mesh1.vertices), len(mesh2.vertices), len(mesh3.vertices)],
        ["Number of faces", len(original.faces),len(mesh1.faces), len(mesh2.faces), len(mesh3.faces)],
        ["Number of intersecting face pairs", len(before_intersections), len(after_intersections1), len(after_intersections2), len(after_intersections3)],
        ["Volume", before_trimesh.volume, after_trimesh1.volume, after_trimesh2.volume, after_trimesh3.volume],
        ["Area", before_trimesh.area, after_trimesh1.area, after_trimesh2.area, after_trimesh3.area],
        ["Mean displacement", "NaN", _evaluate_displacement(original, mesh1), _evaluate_displacement(original, mesh2), _evaluate_displacement(original, mesh3) ]
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

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


def align_submesh_boundary(remaining_mesh, repaired_submesh, tolerance=3):
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

def _evaluate_displacement(mesh1, mesh2):

    original_vertices = mesh1.vertices
    repaired_vertices = mesh2.vertices

    # Build KD-Trees for efficient nearest-neighbor search
    original_tree = cKDTree(original_vertices)
    repaired_tree = cKDTree(repaired_vertices)

    # For each vertex in the original mesh, the distance to the nearest vertex in the repaired mesh.
    distances_original_to_repaired, _ = original_tree.query(repaired_vertices)

    # For each vertex in the repaired mesh, the distance to the nearest vertex in the original mesh.
    distances_repaired_to_original, _ = repaired_tree.query(original_vertices)

    # Combine distances for bidirectional matching
    # Why?
    # - Some vertices in the original mesh may not have a corresponding counterpart in the repaired mesh (e.g., due to merging or removal).
    # - Similarly, new vertices in the repaired mesh may not have counterparts in the original mesh (e.g., due to splitting or addition).
    all_distances = np.concatenate([distances_original_to_repaired, distances_repaired_to_original])

    max_displacement = np.max(all_distances)
    mean_displacement = np.mean(all_distances)

    return mean_displacement


def evaluate_displacement(mesh1, mesh2):

    original_vertices = mesh1.vertices
    repaired_vertices = mesh2.vertices

    # Build KD-Trees for efficient nearest-neighbor search
    original_tree = cKDTree(original_vertices)
    repaired_tree = cKDTree(repaired_vertices)

    # For each vertex in the original mesh, the distance to the nearest vertex in the repaired mesh.
    distances_original_to_repaired, _ = original_tree.query(repaired_vertices)

    # For each vertex in the repaired mesh, the distance to the nearest vertex in the original mesh.
    distances_repaired_to_original, _ = repaired_tree.query(original_vertices)

    # Combine distances for bidirectional matching
    # Why?
    # - Some vertices in the original mesh may not have a corresponding counterpart in the repaired mesh (e.g., due to merging or removal).
    # - Similarly, new vertices in the repaired mesh may not have counterparts in the original mesh (e.g., due to splitting or addition).
    all_distances = np.concatenate([distances_original_to_repaired, distances_repaired_to_original])

    max_displacement = np.max(all_distances)
    mean_displacement = np.mean(all_distances)

    print(f"Max Vertex Displacement: {max_displacement}")
    print(f"Mean Vertex Displacement: {mean_displacement}")

def track_self_intersecting_faces(mesh, intersections):
    """
    Tracks the self-intersecting region's vertices and faces in the original mesh.
    """
    intersecting_faces = set(intersections.flatten())
    intersecting_vertices = np.unique(mesh.faces[list(intersecting_faces)].flatten())
    return intersecting_vertices, intersecting_faces

def map_to_modified_mesh(original_mesh, modified_mesh, intersecting_vertices):
    """
    Maps the intersecting region from the original mesh to the modified mesh.


    e.g.,
    <Original Mesh>
    Vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.5]]
    Faces: [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    Intersecting vertices (detected): [4].

    <Modified Mesh>
    Vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.51, 0.51, 0.51]]

    The original vertex [0.5, 0.5, 0.5] is closest to [0.51, 0.51, 0.51] in the modified mesh.
    So, mapped vertex in modified_mesh is [4]
    """
    # Build a mapping from original vertices to modified vertices
    original_to_modified = {}
    for i, vertex in enumerate(original_mesh.vertices):
        distances = np.linalg.norm(modified_mesh.vertices - vertex, axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < 1e-6:
            original_to_modified[i] = closest_idx

    # Map intersecting vertices to the modified mesh
    mapped_vertices = [original_to_modified[v] for v in intersecting_vertices if v in original_to_modified]
    return np.array(mapped_vertices)

def extract_self_intersecting_region_from_modified(mesh, intersecting_vertices):
    """
    Extracts the submesh corresponding to the intersecting region from the modified mesh.

    e.g.,
    From Step 3, we know the mapped intersecting vertex is [4].
    <Modified Mesh>
    Vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.51, 0.51, 0.51]]
    Faces: [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]].

    Here, we identify faces in the modified mesh that contain any of the intersecting vertices.
    For vertex 4, all four faces ([0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4])
    """
    # Step 1: Identify initial face mask
    face_mask = np.any(np.isin(mesh.faces, intersecting_vertices), axis=1)
    sub_faces = mesh.faces[face_mask]

    # Step 2: Build the initial submesh
    submesh = pymesh.form_mesh(mesh.vertices, sub_faces)

    # Step 3: Find adjacent faces
    from collections import defaultdict

    # Create an edge-to-face map for the entire mesh
    edge_to_faces = defaultdict(list)
    for face_idx, face in enumerate(mesh.faces):
        edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)

    # Collect all edges of the current submesh
    submesh_edges = set()
    for face in sub_faces:
        submesh_edges.update([
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ])

    # Find all adjacent faces to the submesh
    adjacent_faces = set()
    for edge in submesh_edges:
        for face_idx in edge_to_faces[edge]:
            if not face_mask[face_idx]:  # If the face is not already in the submesh
                adjacent_faces.add(face_idx)

    # Update the face mask to include adjacent faces
    updated_face_mask = face_mask.copy()
    updated_face_mask[list(adjacent_faces)] = True

    # Step 4: Rebuild the submesh with the updated face mask
    all_faces = mesh.faces[updated_face_mask]
    updated_submesh = pymesh.form_mesh(mesh.vertices, all_faces)

    # Step 5: Identify and remove outermost faces
    boundary_edges = defaultdict(list)
    for face_idx, face in enumerate(updated_submesh.faces):
        edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ]
        for edge in edges:
            boundary_edges[edge].append(face_idx)

    # Find boundary faces (faces with edges belonging to only one face)
    boundary_faces = set()
    for edge, faces in boundary_edges.items():
        if len(faces) == 1:  # Boundary edge
            boundary_faces.add(faces[0])

    # Create a mask to exclude boundary faces
    non_boundary_face_mask = np.ones(len(updated_submesh.faces), dtype=bool)
    non_boundary_face_mask[list(boundary_faces)] = False

    # Rebuild the submesh without boundary faces
    final_faces = updated_submesh.faces[non_boundary_face_mask]
    final_submesh = pymesh.form_mesh(updated_submesh.vertices, final_faces)

    # Step 6: Update the face mask to reflect the removed outermost faces
    final_face_indices = np.where(updated_face_mask)[0][non_boundary_face_mask]
    final_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    final_face_mask[final_face_indices] = True

    # Clean isolated vertices in the final submesh
    final_submesh, _ = pymesh.remove_isolated_vertices(final_submesh)

    return final_submesh, final_face_mask

def refinement(mesh):
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh, _ = pymesh.remove_degenerated_triangles(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 175.0, 10)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    while len(pymesh.detect_self_intersection(intermediate(mesh))) != 0:
        mesh = pymesh.resolve_self_intersection(intermediate(mesh))
    return mesh

def iterative_repair(mesh, with_rounding=True, precision=11, max_iterations=15):
    """
    Resolves self-intersections in a mesh iteratively.

    Parameters:
        mesh: The input mesh object.
        with_rounding (bool): Enables rounding of vertices for stability. Default is True.
        precision (int): Rounding precision level for vertices.
        max_iterations (int): Maximum number of iterations allowed to resolve intersections.

    Returns:
        mesh: The processed mesh with no self-intersections.
    """
    # Initial rounding of vertices
    if (with_rounding):
        mesh = pymesh.form_mesh(
                np.round(mesh.vertices, precision),
                mesh.faces);
    intersecting_faces = pymesh.detect_self_intersection(mesh);

    # Iterative process to resolve self-intersections
    counter = 0;
    while len(intersecting_faces) > 0 and counter < max_iterations:
        if (with_rounding):
            involved_vertices = np.unique(mesh.faces[intersecting_faces].ravel());

            # Round only the involved vertices
            # Suppose precision = 4. Then,
            # [1.234567, 2.345678, 3.456789] <- One vertex example (x, y, z coords)
            # becomes
            # [1.23, 2.35, 3.46]
            vertices_copy = mesh.vertices.copy()  
            vertices_copy[involved_vertices, :] =\
                    np.round(mesh.vertices[involved_vertices, :],
                            precision//2);
        
            mesh = pymesh.form_mesh(vertices_copy, mesh.faces) 

        mesh = pymesh.resolve_self_intersection(mesh, "igl");
        mesh, __ = pymesh.remove_duplicated_faces(mesh, fins_only=True);
        if (with_rounding):
            mesh = pymesh.form_mesh(
                    np.round(mesh.vertices, precision),
                    mesh.faces);
        mesh = intermediate(mesh) # Reload mesh. Otherwise, the next step fails in some cases.
        intersecting_faces = pymesh.detect_self_intersection(mesh);
        print(len(intersecting_faces))
        counter += 1;

    if len(intersecting_faces) > 0:
        logging.warn("Resolving failed: max iteration reached!");

    return mesh

def remesh(mesh, detail=2.4e-2):

    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    target_len = diag_len * detail
    #print("Target resolution: {} mm".format(target_len))

    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6, preserve_feature=True) # Remove extremely small edges
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                                               preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        #print("#v: {}".format(num_vertices))
        count += 1
        if count > 10: break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.9, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh

def direct_repair(mesh):
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    clean_vertices, clean_faces = pymeshfix.clean_from_arrays(vertices, faces)
    return pymesh.form_mesh(clean_vertices, clean_faces)

def repair_with_local_remesh(mesh):
    intersections = pymesh.detect_self_intersection(mesh)
    intersecting_vertices, intersecting_faces = track_self_intersecting_faces(mesh, intersections)
    outer_hull = pymesh.compute_outer_hull(mesh)
    mapped_vertices = map_to_modified_mesh(mesh, outer_hull, intersecting_vertices)
    submesh, face_mask = extract_self_intersecting_region_from_modified(outer_hull, mapped_vertices)
    remaining_mesh = extract_remaining_mesh(outer_hull, face_mask)

    components = pymesh.separate_mesh(submesh)

    if len(components) == 1:
        repaired_submesh = remesh(submesh)
    else:
        repaired_components = []
        for compoenent in components:
            repaired_component = remesh(compoenent)
            repaired_components.append(repaired_component)
        repaired_submesh = pymesh.merge_meshes(repaired_components)

    aligned_submesh = align_submesh_boundary(remaining_mesh, repaired_submesh)
    repaired_full = replace_submesh_in_original(remaining_mesh, aligned_submesh)
    final = refinement(repaired_full)
    return final