"""
File name: mesh.py
Author: Sebastian Lackner
Version: 1.0
Description: Computation of mesh curvature data
"""

import pymesh
import numpy as np

if __name__ == "__main__":
    quad_mesh = pymesh.load_mesh("in/sample000001_subdiv.ply")
    print(quad_mesh.vertices)
    tri_mesh = pymesh.quad_to_tri(quad_mesh)

    tri_mesh.add_attribute("vertex_gaussian_curvature")
    curvature = tri_mesh.get_attribute("vertex_gaussian_curvature")

    curvature_data = np.hstack((quad_mesh.vertices, np.reshape(curvature, (quad_mesh.num_nodes, 1))))
    print(curvature_data)

    file = open("curvature/sample000001_subdiv.bin", 'wb')
    curvature_data.tofile(file)
    file.close()
