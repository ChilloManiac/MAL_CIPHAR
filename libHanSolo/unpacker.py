COLOR_INDEXES = zip(range(0,1024), range(1024, 2048), range(2048,3072))
INDEX_ORDER = [element for t in COLOR_INDEXES for element in t]
DIR_PATH="./data/"
OUTPUT_PATH=DIR_PATH + "images/"


def unpickle(file): # From https://www.cs.toronto.edu/~kriz/cifar.html
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_value(dict, key):
    return dict[str.encode(key)]


def unpack_file(filename, data, label, is_test):
    from PIL import Image
    test_path = "test/" if is_test else ""
    img = Image.frombytes("RGB", (32,32), data)
    img.save(f'{OUTPUT_PATH}{test_path}{label}/{filename}')


def make_folders():
    import os

    test_path = OUTPUT_PATH + "test/"
    try:
        os.mkdir(OUTPUT_PATH)
        os.mkdir(test_path)
        for k in range(0, 10):
            os.mkdir(OUTPUT_PATH + str(k))
            os.mkdir(test_path + str(k))
    except:
        print ('Error creating data directories, maybe they already existed?')


def unpack_batch(batch, size, is_test=False):
    filenames = [x.decode() for x in get_value(batch, "filenames")]
    decoded_data = get_value(batch, "data")
    labels = get_value(batch, "labels")

    for j in range(0, size):
        data_rgb_ordered = decoded_data[j][INDEX_ORDER]
        unpack_file(filenames[j], data_rgb_ordered, labels[j], is_test)

