from keras.layers import Conv2D, MaxPooling2D, Dense, Input, LeakyReLU, Flatten, Reshape
from keras.models import Sequential
import tensorflow as tf
import keras.backend as K

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=64,
                     kernel_size=(7, 7),
                     strides=(2, 2),
                     input_shape=(448, 448, 3),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same'))

    model.add(Conv2D(filters=192,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same'))

    model.add(Conv2D(filters=128,
                     kernel_size=(1, 1),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=256,
                     kernel_size=(1, 1),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same'))

    for _ in range(0,4):
        model.add(Conv2D(filters=256,
                         kernel_size=(1, 1),
                         padding='same',
                         activation='linear'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(filters=512,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='linear'))
        model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=512,
                     kernel_size=(1, 1),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=1024,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same'))

    for _ in range(0,2):
        model.add(Conv2D(filters=512,
                         kernel_size=(1, 1),
                         padding='same',
                         activation='linear'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(filters=1024,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='linear'))
        model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=1024,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=1024,
                     kernel_size=(3, 3),
                     padding='same',
                     strides=(2,2),
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=1024,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=1024,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Flatten())
    model.add(Dense(4096, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(845, activation='linear'))
    model.add(Reshape((13, 13, 5)))
    model.summary()

    return model


def custom_loss_function(y_true, y_predicted):
    loss = 0
    lambda_coord = 5
    lambda_noobj = 0.5
    c_tru_mask = tf.cast(y_true[:,:,:,4], dtype=tf.bool)
    adjusted_y_true = tf.sigmoid(y_true)
    x_tru = adjusted_y_true[:,:,:,0]
    y_tru = adjusted_y_true[:,:,:,1]
    w_tru = adjusted_y_true[:,:,:,2]
    h_tru = adjusted_y_true[:,:,:,3]
    c_tru = adjusted_y_true[:,:,:,4]

    adjusted_y_ped = tf.sigmoid(y_predicted)
    x_pre = adjusted_y_ped[:,:,:,0]
    y_pre = adjusted_y_ped[:,:,:,1]
    w_pre = adjusted_y_ped[:,:,:,2]
    h_pre = adjusted_y_ped[:,:,:,3]
    c_pre = adjusted_y_ped[:,:,:,4]

    subtracted_x = tf.subtract(x_pre, x_tru)
    subtracted_y = tf.subtract(y_pre, y_tru)
    center_loss = tf.add(tf.square(subtracted_x), tf.square(subtracted_y))
    filtered_center_loss = tf.boolean_mask(center_loss, c_tru_mask)

    rad_w_pre = tf.sqrt(w_pre)
    rad_w_tru = tf.sqrt(w_tru)
    rad_h_pre = tf.sqrt(h_pre)
    rad_h_tru = tf.sqrt(h_tru)
    sub_rad_w = tf.subtract(rad_w_pre, rad_w_tru)
    sub_rad_h = tf.subtract(rad_h_pre, rad_h_tru)
    dim_loss = tf.add(tf.square(sub_rad_w), tf.square(sub_rad_h))
    dim_loss = tf.boolean_mask(dim_loss, c_tru_mask)

    subtracted_c = tf.subtract(c_pre, c_tru)
    class_loss = tf.square(subtracted_c)

    not_c_tru = tf.logical_not(c_tru_mask)
    no_class_loss = tf.boolean_mask(class_loss, not_c_tru)

    loss = lambda_coord*K.sum(filtered_center_loss) + lambda_coord*K.sum(dim_loss) + K.sum(class_loss) + lambda_noobj * K.sum(no_class_loss)
    loss = tf.Print(loss, [y_true], 'true')
    loss = tf.Print(loss, [y_predicted], 'predicted')
    loss = tf.Print(loss, [K.min(filtered_center_loss)], '1')
    loss = tf.Print(loss, [K.min(dim_loss)], '2')
    loss = tf.Print(loss, [K.min(class_loss)], '3')
    loss = tf.Print(loss, [K.min(no_class_loss)], 'center_loss')


    return loss
