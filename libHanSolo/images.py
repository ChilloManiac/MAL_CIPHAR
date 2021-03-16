import numpy as np

def load_image(path):
    from PIL import Image
    return np.asarray(Image.open(path)).reshape(3072)


def get_images_in_label(image_folder_path, label_number):
    import os;
    path = f'{image_folder_path}{label_number}/'
    return np.asarray([load_image(os.path.join(path, file)) for file in os.listdir(path)])


def load_data(image_folder_path):
    X = np.asarray([get_images_in_label(image_folder_path, label) for label in range(0, 10)]).reshape(50000, 3072) 
    y = np.asarray([np.full(5000, i) for i in range(0, 10)]).reshape(50000)
    return (X, y)


def load_test_data(image_folder_path):
    test_path = image_folder_path + "test/"
    X = np.asarray([get_images_in_label(test_path, label) for label in range(0, 10)]).reshape(10000, 3072)
    y = np.asarray([np.full(1000, i) for i in range(0, 10)]).reshape(10000)
    return (X, y)

    

def show_image(image):
    from PIL import Image
    assert image.shape[0] == 3072, "Image should be a 3072 vector"
    Image.fromarray(image.reshape(32, 32, 3)).show()


