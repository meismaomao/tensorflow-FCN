import numpy as np
import scipy.misc as misc

class BatchDataset(object):
    file = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Modified from https://github.com/shekkizh

        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        avaliable option:
        resize = True/False
        resize_size = size of output image - does bilinear resize
        color = True/False
        """

        print('Initializing Batch DataSet Reader')
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_image()

    def _read_image(self):
        self.__channles = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channles = False
        self.annotations = np.array([self._transform(filename['annotation']) for filename in self.files])
        print(np.shape(self.images))
        print(np.shape(self.annotations))

    def _transform(self, filename):

        image = misc.imread(filename)
        if self.__channles and len(image.shape)<3:
            image = np.array([image for _ in range(3)])
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image
        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            self.epochs_completed += 1
            print("**************Epochs completed:" + str(self.epochs_completed) + "***************")

            np.random.shuffle(self.images)
            np.random.shuffle(self.annotations)

            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset

        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return [self.images[index] for index in indexes], [self.annotations[index] for index in indexes]
