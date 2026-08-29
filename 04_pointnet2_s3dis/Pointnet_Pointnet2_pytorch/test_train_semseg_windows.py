import pickle
import unittest

import train_semseg


class TrainSemsegWindowsCompatibilityTest(unittest.TestCase):
    """验证 `train_semseg.py` 在 Windows 多进程下的可序列化要求。"""

    def test_worker_init_fn_is_picklable(self):
        """`worker_init_fn` 必须是模块级函数，便于 Windows spawn 序列化。"""
        worker_init_fn = train_semseg.train_worker_init_fn

        serialized = pickle.dumps(worker_init_fn)

        self.assertGreater(len(serialized), 0)


if __name__ == "__main__":
    unittest.main()
