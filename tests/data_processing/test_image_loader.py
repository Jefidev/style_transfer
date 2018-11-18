import unittest

from src.data_processing.image_loader import ImageLoader


class ImageLoaderChecker(unittest.TestCase):

    def test_init_loader(self):
        imgl = ImageLoader("./tests/loaders/test_img/burning.jpg")

        self.assertEqual(imgl.width, 990)
        self.assertEqual(imgl.height, 589)

    def test_resize_image(self):
        imgl = ImageLoader("./tests/loaders/test_img/burning.jpg")

        imgl.resize(480, 140)

        self.assertEqual(imgl.width, 480)
        self.assertEqual(imgl.height, 140)
