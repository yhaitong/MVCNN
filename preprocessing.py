import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def read_off(file):
    if "OFF" != file.readline().strip():
        raise ("Not a valid OFF header")
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(" ")])
    vertices = [
        [float(s) for s in file.readline().strip().split(" ")] for i in range(n_verts)
    ]
    faces = [
        [int(s) for s in file.readline().strip().split(" ")][1:] for i in range(n_faces)
    ]
    return np.array(vertices), np.array(faces)


def plot_mesh(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces[670:671], shade=True)
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, shade=True
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.plot([1,1], [-5,-5],[1,20], marker='o')
    plt.show()


def fit_to_unit_cube(vertices):
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    scale = max(max_vals - min_vals)
    return (vertices - min_vals) / scale


def plane_projection(point, point_in_screen, normal_vector):
    # point: position vector of a point to be projected onto a plane
    # point_in_plane: position of a point in the scree
    # normal_vector: direction perpendicular to the scree
    p, s, n = point, point_in_screen, normal_vector
    n /= np.linalg.norm(n)
    x = np.dot(s - p, n)
    distance = abs(x)
    p_s = p + x * n
    assert abs(np.dot(p_s - s, n)) < 1e-5
    return distance, p_s


def get_coordinate_in_screen(origin, x_axis, y_axis, normal, point):
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    normal /= np.linalg.norm(normal)
    if abs(np.dot(point - origin, normal)) > 1e-5:
        print("the point provided is not in the screen.")
        return None
    p = point - origin
    return np.dot(p, x_axis), np.dot(p, y_axis)


def generate_views(file_path, output_dir):
    # Read mesh from .off file
    with open(file_path, "r") as file:
        vertices, faces = read_off(file)

        vertices = fit_to_unit_cube(vertices)
        views = {
            "front": (
                np.array([0.0, 0.0, 1.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            ),
            "back": (
                np.array([0.0, 0.0, -1.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            ),
            "left": (
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]),
            ),
            "right": (
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, -1.0]),
                np.array([0.0, 1.0, 0.0]),
            ),
            "top": (
                np.array([0.0, 1.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, -1.0]),
            ),
            "bottom": (
                np.array([0.0, -1.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ),
        }

        for view_name, (_, x_axis, y_axis) in views.items():
            origin = np.array([0.0, 0.0, 2.0])
            normal = np.cross(x_axis, y_axis)
            npixels = 50
            screen = np.ones((npixels + 1, npixels + 1)) * 3.0
            plt.figure(figsize=(8, 6))
            for f in faces:
                # now deal with one face
                nv_f = len(f)
                v, dis, v_s = np.zeros((nv_f, 3)), np.zeros(nv_f), np.zeros((nv_f, 3))
                coord, coord_px = np.zeros((nv_f, 2)), np.zeros(
                    (nv_f, 2), dtype=np.int32
                )
                for i_f, i in enumerate(f):
                    v[i_f] = vertices[i]
                    dis[i_f], v_s[i_f] = plane_projection(v[i_f], origin, normal)
                    coord[i_f] = get_coordinate_in_screen(
                        origin, x_axis, y_axis, normal, v_s[i_f]
                    )
                    coord_px[i_f] = np.clip(
                        np.round(coord[i_f] * npixels).astype(int), 0, npixels
                    )
                    screen[coord_px[i_f][0], coord_px[i_f][1]] = dis[i_f]
                O_px, A_px, B_px = coord_px[1], coord_px[2], coord_px[0]
                OA_px, OB_px = A_px - O_px, B_px - O_px
                if np.abs(np.cross(OA_px, OB_px)) < 1e-5:
                    continue
                for a_x in range(
                    0,
                    (OA_px[0] if OA_px[0] != 0 else 1),
                    np.sign(OA_px[0]) if OA_px[0] != 0 else 1,
                ):
                    for b_x in range(
                        0,
                        (OB_px[0] if OB_px[0] != 0 else 1),
                        np.sign(OB_px[0]) if OB_px[0] != 0 else 1,
                    ):
                        for a_y in range(
                            0,
                            (OA_px[1] if OA_px[1] != 0 else 1),
                            np.sign(OA_px[1]) if OA_px[1] != 0 else 1,
                        ):
                            for b_y in range(
                                0,
                                (OB_px[1] if OB_px[1] != 0 else 1),
                                np.sign(OB_px[1]) if OB_px[1] != 0 else 1,
                            ):
                                d_px = np.array([a_x + b_x, a_y + b_y])
                                # f_A*OA_px+f_B*OB_px = d_px
                                coeff = np.array([OA_px, OB_px])
                                f = np.einsum("i,ij->j", d_px, np.linalg.inv(coeff))
                                newx, newy = O_px[0] + d_px[0], O_px[1] + d_px[1]
                                if (
                                    f[0] + f[1] <= 1.1
                                    and f[0] >= -0.05
                                    and f[1] >= -0.05
                                    and 0 <= newx <= npixels
                                    and 0 <= newy <= npixels
                                ):
                                    screen[newx, newy] = min(
                                        screen[newx, newy],
                                        dis[1]
                                        + f[0] * (dis[2] - dis[1])
                                        + f[1] * (dis[0] - dis[1]),
                                    )
            contour = plt.contourf(screen.transpose(), levels=15, cmap="viridis")
            plt.axis("off")
            plt.savefig(
                os.path.join(
                    output_dir, f"{os.path.basename(file_path)[:-4]}_{view_name}.png"
                ),
                bbox_inches="tight",
            )
            plt.close()


def process_modelnet10(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".off"):
                file_path = os.path.join(root, file)
                output_subdir = root.replace(input_dir, output_dir)
                os.makedirs(output_subdir, exist_ok=True)
                generate_views(file_path, output_subdir)


def main():
    # Example usage
    input_directory = "/Users/haitongyang/Downloads/ModelNet10"
    output_directory = (
        "/Users/haitongyang/Documents/DL/project/MVCNN-PyTorch/data/2dViews"
    )
    process_modelnet10(input_directory, output_directory)


if __name__ == "__main__":
    main()
