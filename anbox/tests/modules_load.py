import sys
import os
import unittest
try:
    import anbox
except:
    # loading some utils
    sys.path.append(os.environ["PWD"] + "/../../")
    import anbox


class ModulesLoading(unittest.TestCase):

    # def setUp(self):
    #    self.seq = range(10)

    def test_loading(self):
        plugins_list = anbox.analysis_loader.get_plugins_list()
        for p in plugins_list:
            algo = anbox.analysis_loader.load(p)
            ret = algo.run()
            self.assertEqual(ret, 0)

    def test_failed_loading(self):
        plugins_list = anbox.analysis_loader.get_plugins_list()
        for p in plugins_list:
            with self.assertRaises(SystemExit):
                anbox.analysis_loader.load(p + "s")

        """
    def test_choice(self):
        #element = random.choice(self.seq)
        self.assertTrue(element in self.seq)

    def test_sample(self):
        with self.assertRaises(ValueError):
            #random.sample(self.seq, 20)
        #for element in random.sample(self.seq, 5):
            self.assertTrue(element in self.seq)
        """


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ModulesLoading)
    unittest.TextTestRunner(verbosity=10).run(suite)
