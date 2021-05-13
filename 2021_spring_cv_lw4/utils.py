import struct


def read_image(image_path):
    with open(image_path, 'rb') as image:
        file_header = get_file_info(image.read(14))
        info_header = get_image_info(image.read(file_header['offset'] - 14))
        pixels = image.read()
        bytes_per_pixel, height, width = get_image_params(info_header)

        return nsplit(pixels, bytes_per_pixel), height, width


def get_file_info(header):
    return {
        'img_type': header[:2].decode(),
        'img_size': convert(header[2:6]),
        'offset': convert(header[10:14]),
    }


def get_image_info(header):
    return {
        'header_size': convert(header[:4]),
        'width': convert(header[4:8]),
        'height': convert(header[8:12]),
        'bits_per_pixel': convert(header[14:16])
    }


def get_image_params(image_info):
    return image_info['bits_per_pixel'] // 8, image_info['height'], image_info['width']


def convert(bytes_text):
    return int(
        struct.unpack(
            'I' if len(bytes_text) > 2 else 'H',
            bytes_text
        )[0]
    )


def nsplit(sequence, length):
    return [
        sequence[i * length:(i + 1) * length]
        for i in range(len(sequence) // length)
    ]


def rgb(p):
    return [
        p[0],
        p[1],
        p[2]
    ]
