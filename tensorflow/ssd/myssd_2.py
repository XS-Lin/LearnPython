import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.layers import concatenate
from keras.models import Model
from keras.engine.topology import InputSpec

import numpy as np
import tensorflow as tf
class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.
    # Arguments
        scale: Default feature scale.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        Same as input
    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    #TODO
        Add possibility to have one scale for all features.
    """
    def __init__(self, scale, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output


class PriorBox(Layer):
    """Generate the prior boxes of designated sizes and aspect ratios.
    # Arguments
        img_size: Size of the input image as tuple (w, h).
        min_size: Minimum box size in pixels.
        max_size: Maximum box size in pixels.
        aspect_ratios: List of aspect ratios of boxes.
        flip: Whether to consider reverse aspect ratios.
        variances: List of variances for x, y, w, h.
        clip: Whether to clip the prior's coordinates
            such that they are within [0, 1].
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        3D tensor with shape:
        (samples, num_boxes, 8)
    # References
        https://arxiv.org/abs/1512.02325
    #TODO
        Add possibility not to have variances.
        Add Theano support
    """
    def __init__(self, img_size=(300.0, 300.0), min_size=30.0, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.waxis = 2
            self.haxis = 1
        else:
            self.waxis = 3
            self.haxis = 2
        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True
        super(PriorBox, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        num_priors_ = len(self.aspect_ratios)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        num_boxes = num_priors_ * layer_width * layer_height
        return input_shape[0], num_boxes, 8

    # support for Keras 2.0
    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        elif hasattr(K, 'int_shape'):
            input_shape = K.int_shape(x)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        img_width = self.img_size[0]
        img_height = self.img_size[1]
        # define prior boxes shapes
        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        # define centers of prior boxes
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        # define xmin, ymin, xmax, ymax of prior boxes
        num_priors_ = len(self.aspect_ratios)
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)
        if self.clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        # define variances
        num_boxes = len(prior_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')
        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        prior_boxes_tensor = K.expand_dims(K.variable(prior_boxes), 0)
        if K.backend() == 'tensorflow':
            pattern = [tf.shape(x)[0], 1, 1]
            prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern)
        elif K.backend() == 'theano':
            #TODO
            pass
        return prior_boxes_tensor

def SSD300v2(input_shape, num_classes=21, featurte_map=None):
    """SSD300 architecture.
    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.
    # References
        https://arxiv.org/abs/1512.02325
    """
    input_layer = Input(shape=input_shape)

    # Block 1
    with tf.name_scope("Block1"):
        conv1_1 = Conv2D(64, (3, 3),
                         name='conv1_1',
                         padding='same',
                         activation='relu')(input_layer)

        conv1_2 = Conv2D(64, (3, 3),
                         name='conv1_2',
                         padding='same',
                         activation='relu')(conv1_1)
        pool1 = MaxPooling2D(name='pool1',
                             pool_size=(2, 2),
                             strides=(2, 2),
                             padding='same', )(conv1_2)

    # Block 2
    with tf.name_scope("Block2"):
        conv2_1 = Conv2D(128, (3, 3),
                         name='conv2_1',
                         padding='same',
                         activation='relu')(pool1)
        conv2_2 = Conv2D(128, (3, 3),
                         name='conv2_2',
                         padding='same',
                         activation='relu')(conv2_1)
        pool2 = MaxPooling2D(name='pool2',
                             pool_size=(2, 2),
                             strides=(2, 2),
                             padding='same')(conv2_2)

    # Block 3
    with tf.name_scope("Block3"):
        conv3_1 = Conv2D(256, (3, 3),
                         name='conv3_1',
                         padding='same',
                         activation='relu')(pool2)
        conv3_2 = Conv2D(256, (3, 3),
                         name='conv3_2',
                         padding='same',
                         activation='relu')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3),
                         name='conv3_3',
                         padding='same',
                         activation='relu')(conv3_2)
        pool3 = MaxPooling2D(name='pool3',
                             pool_size=(2, 2),
                             strides=(2, 2),
                             padding='same')(conv3_3)

    # Block 4
    with tf.name_scope("Block4"):
        conv4_1 = Conv2D(512, (3, 3),
                         name='conv4_1',
                         padding='same',
                         activation='relu')(pool3)
        conv4_2 = Conv2D(512, (3, 3),
                         name='conv4_2',
                         padding='same',
                         activation='relu')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3),
                         name='conv4_3',
                         padding='same',
                         activation='relu')(conv4_2)
        pool4 = MaxPooling2D(name='pool4',
                             pool_size=(2, 2),
                             strides=(2, 2),
                             padding='same')(conv4_3)

    # Block 5
    with tf.name_scope("Block5"):
        conv5_1 = Conv2D(512, (3, 3),
                         name='conv5_1',
                         padding='same',
                         activation='relu')(pool4)
        conv5_2 = Conv2D(512, (3, 3),
                         name='conv5_2',
                         padding='same',
                         activation='relu')(conv5_1)
        conv5_3 = Conv2D(512, (3, 3),
                         name='conv5_3',
                         padding='same',
                         activation='relu')(conv5_2)
        pool5 = MaxPooling2D(name='pool5',
                             pool_size=(2, 2),
                             strides=(1, 1),
                             padding='same')(conv5_3)

    # FC6
    with tf.name_scope("fc6"):
        fc6 = Conv2D(1024, (3, 3),
                     name='fc6',
                     dilation_rate=(6, 6),
                     padding='same',
                     activation='relu'
                     )(pool5)

    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    with tf.name_scope("fc7"):
        fc7 = Conv2D(1024, (1, 1),
                     name='fc7',
                     padding='same',
                     activation='relu'
                     )(fc6)
    # x = Dropout(0.5, name='drop7')(x)

    # Block 6
    with tf.name_scope("Block6"):
        conv6_1 = Conv2D(256, (1, 1),
                         name='conv6_1',
                         padding='same',
                         activation='relu')(fc7)
        conv6_2 = Conv2D(512, (3, 3),
                         name='conv6_2',
                         strides=(2, 2),
                         padding='same',
                         activation='relu')(conv6_1)

    # Block 7
    with tf.name_scope("Block7"):
        conv7_1 = Conv2D(128, (1, 1),
                         name='conv7_1',
                         padding='same',
                         activation='relu')(conv6_2)
        conv7_1z = ZeroPadding2D(name='conv7_1z')(conv7_1)
        conv7_2 = Conv2D(256, (3, 3),
                         name='conv7_2',
                         padding='valid',
                         strides=(2, 2),
                         activation='relu')(conv7_1z)

    # Block 8
    with tf.name_scope("Block8"):
        conv8_1 = Conv2D(128, (1, 1),
                         name='conv8_1',
                         padding='same',
                         activation='relu')(conv7_2)
        conv8_2 = Conv2D(256, (3, 3),
                         name='conv8_2',
                         padding='same',
                         strides=(2, 2),
                         activation='relu')(conv8_1)

    # Last Pool
    with tf.name_scope("LastPool"):
        pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2)

    # Prediction from conv4_3
    num_priors = 3
    img_size = (input_shape[1], input_shape[0])
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)

    with tf.name_scope("conv4_3"):
        conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv4_3)
        conv4_3_norm_mbox_loc = Conv2D(num_priors * 4, (3, 3),
                                       name='conv4_3_norm_mbox_loc',
                                       padding='same')(conv4_3_norm)
        conv4_3_norm_mbox_loc_flat = Flatten(name='conv4_3_norm_mbox_loc_flat')(conv4_3_norm_mbox_loc)
        conv4_3_norm_mbox_conf = Conv2D(num_priors * num_classes, (3, 3),
                                        name=name,
                                        padding='same')(conv4_3_norm)
        conv4_3_norm_mbox_conf_flat = Flatten(name='conv4_3_norm_mbox_conf_flat')(conv4_3_norm_mbox_conf)
        conv4_3_norm_mbox_priorbox = PriorBox(img_size, 30.0,
                                              name='conv4_3_norm_mbox_priorbox',
                                              aspect_ratios=[2],
                                              variances=[0.1, 0.1, 0.2, 0.2])(conv4_3_norm)

    # Prediction from fc7
    num_priors = 6
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    with tf.name_scope("fc7"):
        fc7_mbox_conf = Conv2D(num_priors * num_classes, (3, 3),
                               padding='same',
                               name=name)(fc7)
        fc7_mbox_conf_flat = Flatten(name='fc7_mbox_conf_flat')(fc7_mbox_conf)

        fc7_mbox_loc = Conv2D(num_priors * 4, (3, 3),
                              name='fc7_mbox_loc',
                              padding='same')(fc7)
        fc7_mbox_loc_flat = Flatten(name='fc7_mbox_loc_flat')(fc7_mbox_loc)
        fc7_mbox_priorbox = PriorBox(img_size, 60.0,
                                     name='fc7_mbox_priorbox',
                                     max_size=114.0,
                                     aspect_ratios=[2, 3],
                                     variances=[0.1, 0.1, 0.2, 0.2]
                                 )(fc7)

    # Prediction from conv6_2
    num_priors = 6
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    with tf.name_scope("conv6_2"):
        conv6_2_mbox_conf = Conv2D(num_priors * num_classes, (3, 3),
                                   padding='same',
                                   name=name)(conv6_2)
        conv6_2_mbox_conf_flat = Flatten(name='conv6_2_mbox_conf_flat')(conv6_2_mbox_conf)
        conv6_2_mbox_loc = Conv2D(num_priors * 4, (3, 3,),
                                  name='conv6_2_mbox_loc',
                                  padding='same')(conv6_2)
        conv6_2_mbox_loc_flat = Flatten(name='conv6_2_mbox_loc_flat')(conv6_2_mbox_loc)
        conv6_2_mbox_priorbox = PriorBox(img_size, 114.0,
                                         max_size=168.0,
                                         aspect_ratios=[2, 3],
                                         variances=[0.1, 0.1, 0.2, 0.2],
                                         name='conv6_2_mbox_priorbox')(conv6_2)
    # Prediction from conv7_2
    num_priors = 6
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    with tf.name_scope("conv7_2"):
        conv7_2_mbox_conf = Conv2D(num_priors * num_classes, (3, 3),
                                   padding='same',
                                   name=name)(conv7_2)
        conv7_2_mbox_conf_flat = Flatten(name='conv7_2_mbox_conf_flat')(conv7_2_mbox_conf)
        conv7_2_mbox_loc = Conv2D(num_priors * 4, (3, 3),
                                  padding='same',
                                  name='conv7_2_mbox_loc')(conv7_2)
        conv7_2_mbox_loc_flat = Flatten(name='conv7_2_mbox_loc_flat')(conv7_2_mbox_loc)
        conv7_2_mbox_priorbox = PriorBox(img_size, 168.0,
                                         max_size=222.0,
                                         aspect_ratios=[2, 3],
                                         variances=[0.1, 0.1, 0.2, 0.2],
                                         name='conv7_2_mbox_priorbox')(conv7_2)
    # Prediction from conv8_2
    num_priors = 6
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    with tf.name_scope("conv8_2"):
         conv8_2_mbox_conf = Conv2D(num_priors * num_classes, (3, 3),
                                    padding='same',
                                    name=name)(conv8_2)
         conv8_2_mbox_conf_flat = Flatten(name='conv8_2_mbox_conf_flat')(conv8_2_mbox_conf)
         conv8_2_mbox_loc = Conv2D(num_priors * 4, (3, 3),
                                   padding='same',
                                   name='conv8_2_mbox_loc')(conv8_2)
         conv8_2_mbox_loc_flat = Flatten(name='conv8_2_mbox_loc_flat')(conv8_2_mbox_loc)
         conv8_2_mbox_priorbox = PriorBox(img_size, 222.0,
                                          max_size=276.0,
                                          aspect_ratios=[2, 3],
                                          variances=[0.1, 0.1, 0.2, 0.2],
                                          name='conv8_2_mbox_priorbox')(conv8_2)

    # Prediction from pool6
    num_priors = 6
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    with tf.name_scope("pool6"):
        pool6_mbox_loc_flat = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(pool6)
        pool6_mbox_conf_flat = Dense(num_priors * num_classes, name=name)(pool6)
        pool6_reshaped = Reshape(target_shape,
                                 name='pool6_reshaped')(pool6)
        pool6_mbox_priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
                                       variances=[0.1, 0.1, 0.2, 0.2],
                                       name='pool6_mbox_priorbox')(pool6_reshaped)
    # Gather all predictions
    with tf.name_scope("mbox"):
        mbox_loc = concatenate([conv4_3_norm_mbox_loc_flat,
                                fc7_mbox_loc_flat,
                                conv6_2_mbox_loc_flat,
                                conv7_2_mbox_loc_flat,
                                conv8_2_mbox_loc_flat,
                                pool6_mbox_loc_flat],
                               axis=1, name='mbox_loc')
        mbox_conf = concatenate([conv4_3_norm_mbox_conf_flat,
                                 fc7_mbox_conf_flat,
                                 conv6_2_mbox_conf_flat,
                                 conv7_2_mbox_conf_flat,
                                 conv8_2_mbox_conf_flat,
                                 pool6_mbox_conf_flat],
                                axis=1, name='mbox_conf')
        mbox_priorbox = concatenate([conv4_3_norm_mbox_priorbox,
                                     fc7_mbox_priorbox,
                                     conv6_2_mbox_priorbox,
                                     conv7_2_mbox_priorbox,
                                     conv8_2_mbox_priorbox,
                                     pool6_mbox_priorbox],
                                    axis=1,
                                    name='mbox_priorbox')
        print('{} conv4_3_norm_mbox_loc_flat'.format(conv4_3_norm_mbox_loc_flat._keras_shape))
        print('{} conv4_3_norm_mbox_conf_flat'.format(conv4_3_norm_mbox_conf_flat._keras_shape))
        print('{} conv4_3_norm_mbox_priorbox'.format(conv4_3_norm_mbox_priorbox))
        if hasattr(mbox_loc, '_keras_shape'):
            num_boxes = mbox_loc._keras_shape[-1] // 4
        elif hasattr(mbox_loc, 'int_shape'):
            num_boxes = K.int_shape(mbox_loc)[-1] // 4
        print('{} num_boxes'.format(num_boxes))
        print('{} mbox_loc'.format(mbox_loc._keras_shape))
        print('{} mbox_conf'.format(mbox_conf._keras_shape))
        mbox_loc = Reshape((num_boxes, 4),
                           name='mbox_loc_final')(mbox_loc)
        mbox_conf = Reshape((num_boxes, num_classes),
                            name='mbox_conf_logits')(mbox_conf)
        mbox_conf = Activation('softmax',
                               name='mbox_conf_final')(mbox_conf)
        print('{} locatation'.format(mbox_loc))
        print('{} conf'.format(mbox_conf))
        print('{} priorbox'.format(mbox_priorbox))

    if featurte_map =='conv4_3_norm_mbox_loc_flat':
        return set_return_model(input_layer=input_layer,
                                output_layer=conv4_3_norm_mbox_loc_flat)
    elif featurte_map =='fc7_mbox_loc_flat':
        return set_return_model(input_layer=input_layer,
                                output_layer=fc7_mbox_loc_flat)
    elif featurte_map =='conv4_3_norm_mbox_conf_flat':
        return set_return_model(input_layer=input_layer,
                                output_layer=conv4_3_norm_mbox_conf_flat)
    elif featurte_map =='fc7_mbox_conf_flat':
        return set_return_model(input_layer=input_layer,
                                output_layer=fc7_mbox_conf_flat)
    predictions = concatenate([mbox_loc,
                               mbox_conf,
                               mbox_priorbox],
                              axis=2,
                              name='predictions')
    print('{} predictions'.format(predictions.shape))
    print('{} predictions'.format(predictions))
    model = Model(inputs=input_layer, outputs=predictions)
    return model


def set_return_model(input_layer, output_layer):
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

