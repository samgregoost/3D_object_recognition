import tensorflow as tf
from google.colab import files
import StringIO
import numpy as np
import math

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

uploaded = files.upload()

points_list = []
points = np.array([[0, 0, 0]])
for fn in uploaded.keys():
  
  data = uploaded[fn]
  for line in data.splitlines():
    
#   buf = StringIO.StringIO(data)
#   buf.readline()
#   buf.readline()
    
    
    points_list = [float(s) for s in line.strip().split(' ')]

    points = np.append(points,np.expand_dims(np.array(points_list), axis=0), axis = 0)


print(points)
 

raw_points_init = tf.placeholder(tf.float32, shape=[ None, 3], name="raw_points")
raw_points = tf.expand_dims(raw_points_init, 0,
		                                     name="cn_caps1_output_expanded")

raw_points_ = tf.reshape(raw_points, [-1, 3])
point_count = tf.shape(raw_points)[1]

rotation_matrix_one = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_one")
trasformed_points_one = tf.matmul(raw_points_, rotation_matrix_one, name="trans_point_one")
trasformed_points_one_reshaped = tf.reshape(trasformed_points_one, [-1, point_count, 3], name = "trans_point_one_reshape")

rotation_matrix_two = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_two")
trasformed_points_two = tf.matmul(raw_points_, rotation_matrix_two, name="trans_point_two")
trasformed_points_two_reshaped = tf.reshape(trasformed_points_two, [-1, point_count, 3], name = "trans_point_two_reshape")

rotation_matrix_three = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_three")
trasformed_points_three = tf.matmul(raw_points_, rotation_matrix_three, name="trans_point_three")
trasformed_points_three_reshaped = tf.reshape(trasformed_points_three, [-1, point_count, 3], name = "trans_point_three_reshape")

########################################################################################

trasformed_points_one_reshaped_ = tf.reshape(trasformed_points_one_reshaped, [-1, 3])

point_distance_one = tf.reduce_sum(tf.square(trasformed_points_one_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

scale_metric_one = tf.exp(-point_distance_one*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

scale_metric_tiled_one = tf.tile(scale_metric_one, [1, 3], name="cn_W_tiled")

calibrated_points_one = tf.multiply(scale_metric_tiled_one, trasformed_points_one_reshaped_)



trasformed_points_two_reshaped_ = tf.reshape(trasformed_points_two_reshaped, [-1, 3])

point_distance_two = tf.reduce_sum(tf.square(trasformed_points_two_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

scale_metric_two = tf.exp(-point_distance_two*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

scale_metric_tiled_two = tf.tile(scale_metric_two, [1, 3], name="cn_W_tiled")

calibrated_points_two = tf.multiply(scale_metric_tiled_two, trasformed_points_two_reshaped_)


trasformed_points_three_reshaped_ = tf.reshape(trasformed_points_three_reshaped, [-1, 3])

point_distance_three = tf.reduce_sum(tf.square(trasformed_points_three_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

scale_metric_three = tf.exp(-point_distance_three*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

scale_metric_tiled_three = tf.tile(scale_metric_three, [1, 3], name="cn_W_tiled")

calibrated_points_three = tf.multiply(scale_metric_tiled_three, trasformed_points_three_reshaped_)

calibrated_points_one_corrected_shape = tf.reshape(calibrated_points_one, [-1, point_count, 3])

#########################################################################################


def atan2(y, x):
    angle = tf.select(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle
  

r_one = tf.reduce_sum(tf.square(calibrated_points_one), axis=1, keepdims = True)
theta_one = tf.acos(tf.divide(tf.expand_dims(calibrated_points_one[:,2],1), tf.maximum(r_one,0.001)))
phi_one = tf.atan2(tf.expand_dims(calibrated_points_one[:,1],1),tf.expand_dims(calibrated_points_one[:,0],1))

r_two = tf.reduce_sum(tf.square(calibrated_points_two), axis=1, keepdims = True)
theta_two = tf.acos(tf.divide(tf.expand_dims(calibrated_points_two[:,2],1), tf.maximum(r_two,0.001)))
phi_two = tf.atan2(tf.expand_dims(calibrated_points_two[:,1],1),tf.expand_dims(calibrated_points_two[:,0],1))

r_three = tf.reduce_sum(tf.square(calibrated_points_three ), axis=1, keepdims = True)
theta_three = tf.acos(tf.divide(tf.expand_dims(calibrated_points_three [:,2],1), tf.maximum(r_three,0.001)))
phi_three  = tf.atan2(tf.expand_dims(calibrated_points_three [:,1],1),tf.expand_dims(calibrated_points_three [:,0],1))


r = tf.stack([r_one, r_two, r_three])
theta = tf.stack([theta_one, theta_two, theta_three])
phi = tf.stack([phi_one, phi_two, phi_three])
# polar_coordinates_one = tf.concat([r,theta, phi], axis=1)





# indices = tf.nn.top_k(calibrated_points_one_corrected_shape[:,:, 0], k=point_count).indices

# reordered_calibrated_points_one = tf.gather(calibrated_points_one_corrected_shape, indices, axis=1)

# reordered_calibrated_points_one_corrected_shape = tf.reshape(reordered_calibrated_points_one, [-1, point_count, 3])



# y, idx = tf.unique(tf.squeeze(reordered_calibrated_points_one_corrected_shape[:,:,0], squeeze_dims=[0]))





###########################################################################

y_0_0 = 1.0*math.sqrt(7.0/88.0*1.0) * (tf.zeros(tf.shape(theta)) + 1)
y_0_1 = 1.0*math.sqrt(21.0/88.0*1.0) * tf.cos(theta)
y_1_1 = -1.0 * math.sqrt(21.0/176.0) * (-1.0) * tf.sqrt(1-tf.square(tf.cos(theta)))
y_0_2 = 1.0*math.sqrt(35.0/88.0*1.0) *(1.0/2.0) * (3* tf.square(tf.cos(theta)) - 1)
y_1_2 = -1.0*math.sqrt(35.0/88.0*(1.0/6.0)) * (-1.0) * tf.sqrt(1-tf.square(tf.cos(theta))) * 3.0 * tf.cos(theta)
y_2_2 = 1.0*math.sqrt(35.0/88.0*(1.0/24.0)) * tf.sqrt(1-tf.square(tf.cos(theta))) * 3.0
y_0_3 = 1.0*math.sqrt(49.0/88.0*1.0) * (1.0/2.0) * (5 * tf.pow(tf.cos(theta),3) - 3 * tf.cos(theta))
y_1_3 = -1.0*math.sqrt(49.0/88.0*(1.0/12.0)) * (-1.0) *  (1.0/2.0)*  tf.sqrt(1-tf.square(tf.cos(theta))) * (15 * tf.square(tf.cos(theta)) - 3 )
y_2_3 = 1.0 * math.sqrt(49.0/88.0*(1.0/120.0)) * 15 * tf.cos(theta) * (1-tf.square(tf.cos(theta)))
y_3_3 = -1.0*math.sqrt(49.0/88.0*(1.0/720.0)) * (-1.0) * 15.0 * tf.pow((1-tf.square(tf.cos(theta))),3.0/2.0)

u_0_0 = tf.multiply(y_0_0, tf.cos(0.0))
u_0_1 = tf.multiply(y_0_1, tf.cos(0.0))
u_1_1 = tf.multiply(y_1_1, tf.cos(phi))
u_0_2 = tf.multiply(y_0_2, tf.cos(0.0))
u_1_2 = tf.multiply(y_1_2, tf.cos(phi))
u_2_2 = tf.multiply(y_2_2, tf.cos(2.0*phi))
u_0_3 = tf.multiply(y_0_3, tf.cos(0.0))
u_1_3 = tf.multiply(y_1_3, tf.cos(phi))
u_2_3 = tf.multiply(y_2_3, tf.cos(2.0*phi))
u_3_3 = tf.multiply(y_3_3, tf.cos(3.0*phi))

v_0_0 = tf.multiply(y_0_0, tf.sin(0.0))
v_0_1 = tf.multiply(y_0_1, tf.sin(0.0))
v_1_1 = tf.multiply(y_1_1, tf.sin(phi))
v_0_2 = tf.multiply(y_0_2, tf.sin(0.0))
v_1_2 = tf.multiply(y_1_2, tf.sin(phi))
v_2_2 = tf.multiply(y_2_2, tf.sin(2.0*phi))
v_0_3 = tf.multiply(y_0_3, tf.sin(0.0))
v_1_3 = tf.multiply(y_1_3, tf.sin(phi))
v_2_3 = tf.multiply(y_2_3, tf.sin(2.0*phi))
v_3_3 = tf.multiply(y_3_3, tf.sin(3.0*phi))



U = tf.concat([u_0_0,u_0_1,u_1_1,u_0_2,u_1_2, u_2_2, u_0_3,u_1_3,u_2_3,u_3_3] ,  axis=2)
V = tf.concat([v_0_0,v_0_1,v_1_1,v_0_2,v_1_2, v_2_2, v_0_3,v_1_3,v_2_3,v_3_3] ,  axis=2)

print(U)

init_sigma = 0.01

A_init = tf.random_normal(
		  shape=(3, 10,1),
		  stddev=init_sigma, dtype=tf.float32, name="cn_W_init")
A = tf.Variable(A_init, name="cn_W")


B_init = tf.random_normal(
		  shape=(3, 10,1),
		  stddev=init_sigma, dtype=tf.float32, name="cn_W_init")
B = tf.Variable(B_init, name="cn_W")

estimate = tf.matmul(U,A) + tf.matmul(V,B)

print(B)

loss_estimate = tf.losses.mean_squared_error(r, estimate)

optimizer = tf.train.AdamOptimizer()
grads = optimizer.compute_gradients(loss_estimate)
train = optimizer.apply_gradients(grads)


caps1_raw = tf.reshape(tf.stack([A, B]), [-1, 6, 10],
                       name="caps1_raw")
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
      
caps1_output = squash(caps1_raw, name="caps1_output")

caps2_n_caps = 9
caps2_n_dims = 16


W_init = tf.random_normal(
    shape=(1, 6, 9, 16, 10),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

batch_size = tf.shape(calibrated_points_one_corrected_shape)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")


caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")


caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")
 
#########################################################

raw_weights = tf.zeros([batch_size, 6, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")

# caps1_output_expanded = tf.expand_dims(caps1_output, -1,
#                                        name="caps1_output_expanded")

routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")


caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")




caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, 6, 1, 1, 1],
    name="caps2_output_round_1_tiled")


agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")

routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")


caps2_output = caps2_output_round_2




def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


      
y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth=caps2_n_caps, name="T")


caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 9),
                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 9),
                          name="absent_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")



sess = tf.Session()




# def condition(x, i, index, axis):
#     return tf.logical_and(tf.equal(x[0,i,0], index), tf.equal(x[0,i,2], axis))
  
  
# test = condition(tf.constant([[[10, 20, 30]]]), 0, 20, 30)



# i = tf.constant(0)
# while_condition = lambda i: tf.less(i, tf.shape(reordered_calibrated_points_one_corrected_shape)[1])
# def body(i):
#     x = tf.where(condition(reordered_calibrated_points_one_corrected_shape, i, ), a + 5000, a + b)
#     # increment i
#     return [tf.add(i, 1)]

# # do the loop:
# r = tf.while_loop(while_condition, body, [i])



  

# block_hankel = tf.slice(calibrated_points_one_corrected_shape, [0, 0, 0], [-1,10,-1])
sess.run(tf.global_variables_initializer())


points = sess.run(U, feed_dict = {rotation_matrix_one:[[1, 2, 3], [4, 5, 6], [7, 8, 9]], rotation_matrix_two:[[1, 2, 3], [4, 5, 6], [7, 8, 9]], rotation_matrix_three:[[1, 2, 3], [4, 5, 6], [7, 8, 9]], raw_points_init:points})




print(points)

