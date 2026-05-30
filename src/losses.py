import tensorflow as tf


def weighted_focal_loss(alpha=0.25, gamma=2.0):

    def loss(y_true, y_pred):

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        cross_entropy = -y_true * tf.math.log(y_pred)

        focal_weight = alpha * tf.pow(
            1 - y_pred,
            gamma
        )

        loss_value = focal_weight * cross_entropy

        return tf.reduce_mean(
            tf.reduce_sum(loss_value, axis=1)
        )

    return loss