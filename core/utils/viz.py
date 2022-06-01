import open3d as o3d
import numpy as np
import ipdb
from plyfile import PlyElement, PlyData
import trimesh
import time
import seaborn as sns
import matplotlib.pyplot as plt


def make_o3d_pcd(xyz, colors=None, big_size=False, name='pcd1'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if big_size:
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 10.0
        return {'name': name, 'geometry': pcd, 'material': mat}
    else:
        return pcd


def custom_draw_geometry_with_key_callback(pcd, save_path):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.axis('off')
        plt.imshow(np.asarray(image)[100:900, 450:1400]) #
        fig = plt.gcf()
        fig.set_size_inches(7.0/3, 7.0/3) #dpi = 300, output = 700*700 pixels
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0, 0)
        #plt.show()

        fig.savefig(save_path, bbox_inches='tight', dpi=400, pad_inches = 0)
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks(pcd, key_to_callback)


def viz_pc_keypoint(pc, keypoints, kp_radius=0.3, save_path=None, color_v=[0.792, 0.792, 0.792]):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    #colors = np.ones_like(pc)
    #colors *= 202/255 #202
    colors = np.zeros_like(pc)
    colors[:, 0] = color_v[0]
    colors[:, 1] = color_v[1]
    colors[:, 2] = color_v[2]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    palette = sns.color_palette("bright", 25)  # create color palette
    
    # draw pcd and keypoints
    mesh_spheres = []
    for i in range(keypoints.shape[0]):
        kp = keypoints[i]
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=kp_radius)
        mesh_sphere.translate(kp)
        mesh_sphere.paint_uniform_color([1.0, 0, 0]) #kp['semantic_id']
        mesh_spheres.append(mesh_sphere)
    
    print('visualizing point cloud with keypoints highlighted')
    #o3d.visualization.draw_geometries()
    custom_draw_geometry_with_key_callback([pcd, *mesh_spheres], save_path)


def viz_mesh_keypoint(textured_mesh, keypoints):
    # draw mesh and keypoints
    #textured_mesh = o3d.io.read_triangle_mesh('models/{}/{}/models/model_normalized.obj'.format(class_id, model_id))
    textured_mesh.compute_vertex_normals()
    vertices = np.array(textured_mesh.vertices)
    faces = np.array(textured_mesh.triangles)
    
    mesh_spheres = []
    for i in range(keypoints.shape[0]):
        kp = keypoints[i]
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        face_coords = vertices[faces[kp['mesh_info']['face_index']]]
        mesh_sphere.translate(face_coords.T @ kp['mesh_info']['face_uv'])
        mesh_sphere.paint_uniform_color([1.0, 0, 0])
        mesh_spheres.append(mesh_sphere)
        
    print('visualizing mesh with keypoints highlighted')
    o3d.visualization.draw_geometries([textured_mesh, *mesh_spheres])


def viz_ply_keypoint(textured_mesh, keypoints, radius, color_v=[0.792, 0.792, 0.792]):
    # draw ply and keypoints
    #textured_mesh = o3d.io.read_triangle_mesh('models/{}/{}/models/model_normalized.ply'.format(class_id, model_id))
    textured_mesh.compute_vertex_normals()
    vertices = np.array(textured_mesh.vertices)
    faces = np.array(textured_mesh.triangles)
    
    mesh_spheres = []
    for i in range(keypoints.shape[0]):
        kp = keypoints[i]
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        #face_coords = vertices[faces[kp['mesh_info']['face_index']]]
        #mesh_sphere.translate(face_coords.T @ kp['mesh_info']['face_uv'])
        mesh_sphere.translate(kp)
        mesh_sphere.paint_uniform_color(color_v)
        mesh_spheres.append(mesh_sphere)
        
    print('visualizing ply mesh with keypoints highlighted')
    o3d.visualization.draw_geometries([textured_mesh, *mesh_spheres])


def viz_color_ply(textured_mesh, save_path, color_v=[0.792, 0.792, 0.792]):
    textured_mesh.paint_uniform_color(color_v)
    custom_draw_geometry_with_key_callback([textured_mesh], save_path)


def create_colormap(VERT):
    """
    Creates a uniform color map on a mesh
    Args:
        VERT (Nx3 ndarray): The vertices of the object to plot
    Returns:
        Nx3: The RGB colors per point on the mesh
    """
    VERT = np.double(VERT)
    minx = np.min(VERT[:, 0])
    miny = np.min(VERT[:, 1])
    minz = np.min(VERT[:, 2])
    maxx = np.max(VERT[:, 0])
    maxy = np.max(VERT[:, 1])
    maxz = np.max(VERT[:, 2])
    colors = np.stack([((VERT[:, 0] - minx) / (maxx - minx)), ((VERT[:, 1] - miny) /
                                                               (maxy - miny)), ((VERT[:, 2] - minz) / (maxz - minz))]).transpose()
    return colors


def saliency_pcd(xyz, saliency=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    colors = np.zeros((saliency.shape[0], 3))
    colors[:, 0] = saliency
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def export_pointcloud(vertices, out_file, as_text=True):
    assert(vertices.shape[1] == 3)
    vertices = vertices.astype(np.float32)
    vertices = np.ascontiguousarray(vertices)
    vector_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = vertices.view(dtype=vector_dtype).flatten()
    plyel = PlyElement.describe(vertices, 'vertex')
    plydata = PlyData([plyel], text=as_text)
    plydata.write(out_file)


def extract_mesh(occ_hat, c=None, padding=0.125, threshold=0.2, stats_dict=dict()):
    from core.nets.utils import libmcubes
    ''' Extracts the mesh from the predicted occupancy grid.

    Args:
        occ_hat (tensor): value grid of occupancies
        c (tensor): encoded feature volumes
        stats_dict (dict): stats dictionary
    '''
    # Some short hands
    n_x, n_y, n_z = occ_hat.shape
    box_size = 1 + padding
    threshold = np.log(threshold) - np.log(1. - threshold)
    # Make sure that mesh is watertight
    t0 = time.time()
    
    occ_hat_padded = np.pad(
        occ_hat, 1, 'constant', constant_values=-1e6)
    vertices, triangles = libmcubes.marching_cubes(
        occ_hat_padded, threshold)
    stats_dict['time (marching cubes)'] = time.time() - t0
    # Strange behaviour in libmcubes: vertices are shifted by 0.5
    vertices -= 0.5
    # # Undo padding
    vertices -= 1
    
    # Normalize to bounding box
    vertices /= np.array([n_x-1, n_y-1, n_z-1])
    vertices = box_size * (vertices - 0.5)
    
    # Estimate normals if needed
    # if with_normals and not vertices.shape[0] == 0:
    #     t0 = time.time()
    #     normals = self.estimate_normals(vertices, c)
    #     stats_dict['time (normals)'] = time.time() - t0

    # else:
    normals = None

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles,
                            vertex_normals=normals,
                            process=False)

    
    # Directly return if mesh is empty
    if vertices.shape[0] == 0:
        return mesh

    # TODO: normals are lost here
    # if self.simplify_nfaces is not None:
    #     t0 = time.time()
    #     mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
    #     stats_dict['time (simplify)'] = time.time() - t0

    # Refine mesh
    # if self.refinement_step > 0:
    #     t0 = time.time()
    #     self.refine_mesh(mesh, occ_hat, c)
    #     stats_dict['time (refine)'] = time.time() - t0

    return mesh


def check_transform(pts1, pts2, transform_matrix=None):
    if transform_matrix is None:
        pts2_n = pts2
    else:
        pts2_n = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), 1)
        pts2_n = np.matmul(transform_matrix, pts2_n.T).T #20, 3
    colors1 = np.zeros_like(pts1)
    colors1[:, 0] = 1
    colors2 = np.zeros_like(pts2_n)
    colors2[:, 1] = 1
    input_pcd_show = make_o3d_pcd(pts1, colors1)
    pts2_show = make_o3d_pcd(pts2_n, colors2)
    o3d.visualization.draw_geometries([input_pcd_show, pts2_show])


def check_kp_transform(kp1, kp2, pts1, pts2, transform_matrix):
    pts2_n = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), 1)
    pts2_n = np.matmul(transform_matrix, pts2_n.T).T #20, 3

    kp2_n = np.concatenate((kp2, np.ones((kp2.shape[0], 1))), 1)
    kp2_n = np.matmul(transform_matrix, kp2_n.T).T #20, 3

    colors1 = np.zeros_like(pts1)
    colors2 = np.zeros_like(pts2_n)
    colors2[:, 2] = 0.5

    pts1_show = make_o3d_pcd(pts1, colors1)
    pts2_show = make_o3d_pcd(pts2_n, colors2)

    colors3 = np.zeros_like(kp1)
    colors3[:, 0] = 1

    colors4 = np.zeros_like(kp2_n)
    colors4[:, 1] = 1

    kp1_show = make_o3d_pcd(kp1, colors3)
    kp2_show = make_o3d_pcd(kp2_n, colors4)

    o3d.visualization.draw_geometries([pts1_show, pts2_show, kp1_show, kp2_show])

    o3d.visualization.draw_geometries([kp1_show, kp2_show])