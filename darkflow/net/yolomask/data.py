from .predict import resize_input
import tensorflow as tf
import pickle
import numpy as np
import os

def _decode(self, serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/source_id': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            # 'image/depth': tf.FixedLenFeature([], tf.int64),

            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/class/label': tf.VarLenFeature(tf.int64),

            'image/object/mask': tf.FixedLenFeature([], tf.string),
        })

    source_id = features['image/source_id']

    # Cast height
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    depth = tf.constant(3) # tf.cast(features['image/depth'], tf.int32)

    image_shape = tf.stack([height, width, depth])
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.reshape(image, image_shape)
    image = self.resize_input(image)
    
    # Bounding Boxes
    xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'], default_value=0)
    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'], default_value=0)
    xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'], default_value=0)
    ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'], default_value=0)
    bbxs = tf.concat([[xmin], [ymin], [xmax], [ymax]], 0)
    bbxs = tf.transpose(bbxs)
    bbxlabels = tf.cast(tf.sparse_tensor_to_dense(features['image/object/class/label'], default_value=0), tf.int32)

    # Need format {inp_ph: img, 'meta', :(source_id, (w, h, labels, bbox))) }
    # bbox format is [ (xmin, ymin, xmax, ymax) ]
    contents = (image, source_id, (width, height, (bbxlabels, bbxs)))
    return contents

def _batch(self, inp_ph, chunk):
    meta = self.meta
    labels = meta['labels']
    
    H, W, _ = meta['out_size']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']

    img = chunk[0]
    w, h, (lab, bbox) = chunk[2]

    w = tf.cast(w, tf.float32)
    h = tf.cast(h, tf.float32)

    # Calculate regression target
    cellx = 1. * w / W
    celly = 1. * h / H    
    centerx = .5 * (bbox[:, 0] + bbox[:, 2]) #xmin, xmax
    centery = .5 * (bbox[:, 1] + bbox[:, 3]) #ymin, ymax
    cx = centerx / cellx 
    cy = centery / celly
    bboxt = tf.transpose(bbox)
    # some condition if cx >= W or cy >= H: return None, None
    newxmax = tf.sqrt((bboxt[2] - bboxt[0]) / w) #normalised xdiff
    newymin = tf.sqrt((bboxt[3] - bboxt[1]) / h) #normalised ydiff
    newxmin = cx - tf.floor(cx)
    newymax = cy - tf.floor(cy)
    # bboxt = tf.assign(tf.stack([newxmin, newymin, newxmax, newymax])
    reg = tf.cast(tf.floor(cy) * W + tf.floor(cx), tf.int32)

    ph = self.placeholders
    # Calculate placeholders' values
    regprobs_np = np.zeros((B, C))
    # regprobs_np[:, labels.index(lab[0])] = 1.
    regprobs = tf.constant(regprobs_np)
    regproid = tf.ones((B, C))
    regcoords = tf.stack([bbox] * B)
    xleft = tf.square(bboxt[0] - bboxt[2]) * .5 * W
    yup = tf.square(bboxt[1] - bboxt[3]) * .5 * H
    xright = tf.square(bboxt[0] + bboxt[2]) * .5 * W
    ybot = tf.square(bboxt[1] + bboxt[3]) * .5 * H
    confs = tf.ones(B)

    # Finalise variable values
    # TODO
    
    probs = tf.zeros((H*W, B, C))
    confs = tf.zeros((H*W, B))
    coord = tf.zeros((H*W, B, 4))
    proid = tf.zeros((H*W, B, C))
    prear = tf.zeros((H*W, 4))

    # Finalise the placeholders' values
    upleft   = tf.expand_dims(prear[:,0:2], 1)
    botright = tf.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft; 
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = tf.stack([upleft] * B, 1)
    botright = tf.stack([botright] * B, 1)
    areas = tf.stack([area] * B, 1)

    print('hello1')

    # value for placeholder at loss layer 
    loss_feed_val = {
        'probs': probs, 'confs': confs, 
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft, 
        'botright': botright
    }

    print('hello2')

    # Place new variable values into output dataset
    newdataset = {ph[key]: loss_feed_val[key] for key in ph} # dictionary of ph
    newdataset[inp_ph] = img #img

    return (newdataset)

def parse(self):
    tfrecord_path = self.FLAGS.tfrecord
    all_tfrecords = os.listdir(tfrecord_path)
    all_tfrecords = [i for i in all_tfrecords if i.endswith('tfrecord')]
    if not all_tfrecords:
        msg = 'Failed to find any tfrecords in {} .'
        exit('Error: {}'.format(msg.format(tfrecord_path)))
    
    dataset = tf.data.TFRecordDataset(all_tfrecords)
    dataset = dataset.map(self._decode)
    return dataset

def input_pipeline(self, inp_ph):
    num_batch = self.FLAGS.batch
    dataset = self.parse()
    dataset = dataset.shuffle(buffer_size=10000)

    # Batch dataset
    dataset = dataset.map(lambda *d: self._batch(inp_ph, d))
    dataset = dataset.batch(num_batch)
     
    iterator = dataset.make_one_shot_iterator()
    self.next_batch = iterator.get_next()

def shuffle(self):
    print('Shuffling TFRecord dataset')
    return self.next_batch
