import os
import sys
import random
import glob
import pickle
import zipfile
import tarfile
import urllib

DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'

def maybe_download_and_extract(dir_path, model_url, is_zipfile=False, is_tarfile=False):
    """
    Modified implementation from tensorflow/model/cifar10/input_data
    :param dir_path:
    :param model_url:
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = model_url.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Download %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        file_path, msg = urllib.urlretrieve(model_url, filepath, reporthook=_progress)
        print(msg)
        print('\n')
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                # zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)
        elif is_tarfile:
            tarfile.open(file_path, 'r:gz').extractall(dir_path)


def create_image_list(image_dir):
    if not os.path.isfile(image_dir):
        print("Image_directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "image", directory, '*.jpg')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            # raise Exception('file _list is None!')
            print('No files found')
            return None
        else:
            for f in file_list:
                filename = os.path.splitext(f.split('/')[-1])[0]
                annotation_file = os.path.join(image_dir, 'annotations', directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print('annotation file not found for %s - Skipping'%filename)

        random.shuffle(image_list[directory])
        num_of_images = len(image_list[directory])
        print('No. of %s files: %d' % (directory, num_of_images))

    return image_list

def read_dataset(data_dir):
    pickle_filename = 'MITScenceParsing.pickle'
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        result = create_image_list(os.path.join(data_dir, SceneParsing_folder))
        print("Pickling...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print('Found pickle file!')

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result
    return training_records, validation_records

#test programme
# def main(argv=None):
#     read_dataset('dataset_file')
#     # SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
#     # print(DATA_URL.split("/"))
#     # print(SceneParsing_folder)
#
# if __name__ == '__main__':
#     main()