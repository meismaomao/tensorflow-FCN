import tensorflow as tf
import numpy as np
import utils
import read_data as scene_parsing
import datetime
import batch_dataset_reader as dataset

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
IMAGE_SIZE = 224
NUM_OF_CLASSESS = 151
LEARNING_RATE = 1e-4
MAX_ITERATION = int(1e5)
BATCH_SIZE = 4
DATA_DIR = './dataset_file'

# load the VGG model and add weights, bias etc params
def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_1', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            #kernels are [width, height, in_channles, out_channles]
            #tensorflow are [height, width, in channles, out_channles]
            kernels = utils.get_variable(
                np.transpose(kernels, (1, 0, 2, 3)), name=name+'_w')
            bias = utils.get_variable(bias.reshape(-1), name=name+'_b')
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
    return net

def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image:
    :param keep_prob:
    :return:
    """
    print('setting up vgg model initialized params')
    model_data = utils.get_model_data("data", MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.name_scope('inference'):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net['conv5_3']

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weights_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name='b6')
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name='relu6')

        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weights_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")

        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weights_variable([1, 1, 4096, NUM_OF_CLASSESS], name='W8')
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

        #unsampling to actual image size
        deconv_shape1 = image_net['pool4'].get_shape()
        W_t1 = utils.weights_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name='W_t1')
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net['pool4']))
        fuse_1 = tf.add(conv_t1, image_net['pool4'], name='fuse_1')

        deconv_shape2 = image_net['pool3'].get_shape()
        W_t2 = utils.weights_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name='W_t2')
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net['pool3']))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        output_shape = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weights_variable([7, 7, NUM_OF_CLASSESS, deconv_shape2[3].value], name='W_t3')
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=output_shape)

        annotation_pre = tf.argmax(conv_t3, dimension=3, name='prediction')

        return tf.expand_dims(annotation_pre, dim=3), conv_t3

def Train_op(loss, var_list):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = optimizer.compute_gradients(loss, var_list)
    for grad, var in grads:
        utils.add_grad_summary(grad, var)
    return optimizer.apply_gradients(grads)

def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name='keep_probability')
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input_image')
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='annotation')

    pre_annotation, logits = inference(image, keep_prob=keep_probability)
    tf.summary.image('input_image', image, max_out_puts=2)
    tf.summary.image('ground_truth', annotation, max_outputs=2)
    tf.summary.image('pre_annotation', pre_annotation, max_outputs=2)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits\
        (logits=logits, labels=tf.squeeze(annotation, aqueeze_dims=[3]), name='entropy')
    tf.summary.scalar('entropy', loss)

    var_list = tf.trainable_variables()

    train_op = Train_op(loss, var_list)

    print('Setting up summary op...')
    summary_op = tf.summary.merge_all()

    print('Setting up read data set')
    train_records, valid_records = scene_parsing.read_dataset(DATA_DIR)
    print(len(train_records))
    print(len(valid_records))

    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    train_dataset_reader = dataset.BatchDataset(train_records, image_options)
    valid_dataset_reader = dataset.BatchDataset(valid_records, image_options)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.global_variables_initializer()
        summary_writer = tf.summary.FileWriter('./log_dir', sess.graph)

        for itr in range(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(BATCH_SIZE)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print('Step: %d, Train_loss: %g' % (itr, train_loss))
                # The add_summary function args of global_step(str) is optional.
                summary_writer.add_summary(summary_str, str)

            if itr % 100 == 0:
                gpre_annotation = sess.run(pre_annotation, feed_dict=feed_dict)
                utils.save_image_pre_annotation(gpre_annotation, train_images, train_annotations)

            if itr % 500 == 0:
                valid_images, valid_annotations = valid_dataset_reader.get_random_batch(BATCH_SIZE)
                valid_loss = sess.run(loss, feed_dict={image: valid_images,
                                                       annotation: valid_annotations, keep_probability: 1.0})
                print("%s ---> Valid loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, './models', global_step=itr)

if __name__ == "__main__":
    main()


























