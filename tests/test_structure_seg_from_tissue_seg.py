import numpy as np
import timeit
from src.multi_atlas.multi_atlas_fusion import get_structure_seg_from_tissue_seg
import unittest

FIXED_INPUT = False

if FIXED_INPUT:
    tiss_seg = np.array([[[1, 0], [3, 3]], [[2, 2], [3, 3]]]).transpose(2, 1, 0)
    lab_probs = np.array([[[[.4, .6], [.2, .4]], [[.4, .6], [.2, .4]]], [[[.3, .2], [.5, .1]], [[.3, .2], [.5, .1]]], [[[.3, .2], [.3, .5]], [[.3, .2], [.3, .5]]]] ).transpose(3, 2, 1, 0)
    tissue_dict = {0: [0], 1: [1, 2, 3], 2: [3], 3:[3]}

else:
    size = np.array([200, 100, 10])
    nb_structures = 160
    nb_tissues = 7
    tiss_seg = np.random.randint(0, nb_tissues+1, size=size, dtype=np.uint16)
    lab_probs = np.random.rand(*size, nb_structures)
    # make sure lab_probs sums to 1 along last dimension
    lab_probs = lab_probs / np.sum(lab_probs, axis=-1, keepdims=True)

    # create random tissue_dict which assignes each label to a random subset of at most 5 tissues
    tissue_dict = {}
    for l in range(nb_structures):
        tissue_dict[l] = np.random.choice(nb_tissues, size=np.random.randint(1, 5), replace=False)

print(tiss_seg.shape)
print(lab_probs.shape)


# check that values add up to 1 along label dimension, allowing for small numerical errors
assert(np.sum(lab_probs, axis=-1) - 1 < 1e-6).all(), "Probabilities do not sum to 1 along label dimension, but instead to {}".format(np.sum(lab_probs, axis=-1))

## compare with nested loop method
def nested_loop_method(tiss_seg, lab_probs, tissue_dict):
    structure_seg = np.zeros_like(tiss_seg)
    sss = structure_seg.shape
    probs_idx_sorted = np.argsort(lab_probs, axis=-1)[...,::-1]
    for i in range(sss[0]):
        #print(i)
        for j in range(sss[1]):
            for k in range(sss[2]):

                probs_idx_sorted_ijk = probs_idx_sorted[i, j, k]

                for probs_idx in probs_idx_sorted_ijk:
                    if tiss_seg[i, j, k] in tissue_dict[probs_idx]:
                        structure_seg[i, j, k] = probs_idx
                        break

    return structure_seg

class YourTestCase(unittest.TestCase):
    def test_check_results_and_compare_runtimes(self):
        # method 1
        start1 = timeit.default_timer()
        structure_seg1 = get_structure_seg_from_tissue_seg(tiss_seg, lab_probs, tissue_dict)
        stop1 = timeit.default_timer()
        print('\nTime get_structure_seg_from_tissue_seg: ', "{:.5f}".format(stop1 - start1), " seconds")

        # method 2
        start = timeit.default_timer()
        structure_seg2 = nested_loop_method(tiss_seg, lab_probs, tissue_dict)
        stop = timeit.default_timer()
        print('Time nested_loop_method: ', stop - start)


        # check if outputs are the same
        self.assertTrue(np.alltrue(structure_seg1 == structure_seg2)), "Outputs are not the same"

        # print ratio of runtimes
        print('Ratio of runtimes: ', (stop1 - start1) / (stop - start))

if __name__ == '__main__':
    unittest.main()


