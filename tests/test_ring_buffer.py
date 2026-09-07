import os
import sys
import numpy as np
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.ring_buffer import RingBuffer

class TestRingBuffer(unittest.TestCase):
    def test_append_and_get(self):
        rb = RingBuffer(capacity=10)
        
        # Test appending small chunks
        rb.append(np.array([1, 2, 3]))
        res = rb.get_latest(3)
        np.testing.assert_array_equal(res, [1, 2, 3])
        
        rb.append(np.array([4, 5, 6]))
        res = rb.get_latest(6)
        np.testing.assert_array_equal(res, [1, 2, 3, 4, 5, 6])
        
        # Test overflowing the buffer
        rb.append(np.array([7, 8, 9, 10, 11]))
        # Buffer should now hold: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        res = rb.get_latest(10)
        np.testing.assert_array_equal(res, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        
        # Test getting smaller slice
        res = rb.get_latest(4)
        np.testing.assert_array_equal(res, [8, 9, 10, 11])
        
        # Test appending larger than capacity
        rb.append(np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]))
        res = rb.get_latest(10)
        np.testing.assert_array_equal(res, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

if __name__ == '__main__':
    unittest.main()
