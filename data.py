import util
import numpy as np
import gzip
import struct

class MNIST:
    urls = {
        'train': {
            'image': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'label': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            },
        'test': {
            'image': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'label': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
            }
    }

    def __init__(self, split, one_hot=False):
        assert(split in MNIST.urls)
        self.one_hot = one_hot

        # check if downloads exist, and download otherwise
        image_file = util.download(MNIST.urls[split]['image'])
        label_file = util.download(MNIST.urls[split]['label'])

        self.x, self.y = self._load(image_file, label_file)

    def __getitem__(self, key):
        return self.x[key], self.y[key]

    def __len__(self):
        return len(self.x)

    def _read_idx(self, filepath, num_dims):

        base_magic_num = 2048
        with gzip.GzipFile(filepath) as f:
            magic_num = struct.unpack('>I', f.read(4))[0]
            expected_magic_num = base_magic_num + num_dims
            if magic_num != expected_magic_num:
                raise ValueError('Incorrect MNIST magic number (expected '
                                 '{}, got {})'
                                 .format(expected_magic_num, magic_num))
            dims = struct.unpack('>' + 'I' * num_dims,
                                 f.read(4 * num_dims))

            buf = f.read(np.prod(dims))
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(*dims)
            return data


    def _load(self, image_file, label_file):
        # Must initialize tf.GraphKeys.TABLE_INITIALIZERS
        # sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))

        X = self._read_idx(image_file, 3).reshape([-1, 28, 28, 1]).astype(np.float32) / 255
        y = self._read_idx(label_file, 1).astype(np.int64)

        if self.one_hot:
            y_ = np.zeros((X.shape[0], 10))
            y_[np.arange(y_.shape[0]), y] = 1
            y = y_

        return X, y


class DataLoader:
    def __init__(self, dataset, batch_size, repeat=False, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.idx = None
        self.step = 0

    def reset(self):
        self.idx = np.arange(len(self.dataset))
        self.step = 0
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):

        # If we're past the last batch
        if ((self.step+self.batch_size) >= len(self.dataset) and self.drop_last) or self.step >= len(self.dataset):
            if self.repeat:
                self.reset()
            else:
                raise StopIteration

        data = self.dataset[self.idx[self.step:self.step+self.batch_size]]
        self.step += self.batch_size
        return data

if __name__ == '__main__':
    mnist = MNIST('train')

    dataloader = DataLoader(mnist, batch_size=33, drop_last=True, shuffle=False, repeat=True)

    for i, (x, y) in enumerate(dataloader):
        print(i, x.shape, np.mean(x), y)