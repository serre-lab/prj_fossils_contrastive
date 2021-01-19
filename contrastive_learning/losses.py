import tensorflow as tf

LARGE_NUM = 1e9


def get_contrastive_loss(temperature=1.0):
   
    def contrastive_loss(projection):

        projection = tf.math.l2_normalize(projection, -1)
        
        proj1, proj2 = tf.split(projection, 2, 0)
        batch_size = tf.shape(proj1)[0]

        labels = tf.one_hot(tf.range(batch_size), batch_size * 2) #[[1,0,0, 0, 0, 0]
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(proj1, proj1, transpose_b=True) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(proj2, proj2, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(proj1, proj2, transpose_b=True) / temperature
        logits_ba = tf.matmul(proj2, proj1, transpose_b=True) / temperature
                                                        
        loss_a = tf.nn.softmax_cross_entropy_with_logits(
                                                        labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.nn.softmax_cross_entropy_with_logits(
                                                        labels, tf.concat([logits_ba, logits_bb], 1))
        loss = tf.reduce_mean(loss_a + loss_b)

        #return loss, logits_ab, labels
        return loss
    
    return contrastive_loss
