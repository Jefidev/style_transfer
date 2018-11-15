import argparse

from src.loaders.image_loader import ImageLoader


def main_process():
    source = ImageLoader("data/oim.jpg")
    style = ImageLoader("data/magritte.jpg")

    style.resize(source.width, source.height)


if __name__ == "__main__":
    main_process()
