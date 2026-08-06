from ur_toolbox.transformation.pose import translation_rotation_2_array, translation_rotation_2_matrix, pose_array_2_matrix, pose_matrix_2_array
import unittest
import numpy as np

class transformation_Tests(unittest.TestCase):
    def test_aux_calc_projection_length(self):
        self.assertAlmostEqual(np.sum(translation_rotation_2_array(np.array((1,2,3.0)), np.eye(3,dtype = float))), 6)
        m = translation_rotation_2_matrix(np.array((1,2,3.0)), np.eye(3,dtype = float))
        self.assertAlmostEqual(np.sum(m), 10)
        self.assertAlmostEqual(np.mean(m), 10 /16.0)
        p_matrix = pose_array_2_matrix(np.array([1,2,3,0,0,0]))
        self.assertAlmostEqual(np.sum(p_matrix), 10)
        self.assertAlmostEqual(np.mean(p_matrix), 10 / 16.0)
        p_array=pose_matrix_2_array(m)
        self.assertAlmostEqual(np.sum(p_array), 6)
        self.assertAlmostEqual(len(p_array), 6)