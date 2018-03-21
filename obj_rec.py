import numpy as np
import math
import os
import tensorflow as tf
import glob
    
print(tf.__version__)


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 

raw_points_init = tf.placeholder(tf.float32, shape=[ None, 3], name="raw_points")

centered_points = tf.subtract(raw_points_init, tf.reduce_mean(raw_points_init, axis = 0, keepdims = True))

centered_points_expanded = tf.expand_dims(centered_points, 0,
		                                     name="cn_caps1_output_expanded")

adjoint_mat = tf.matmul(tf.transpose(centered_points_expanded, [0,2,1]), centered_points_expanded)

e,ev = tf.self_adjoint_eig(adjoint_mat, name="eigendata")



normal_vec = ev[:,:,0]
normalized_normal_vec = tf.nn.l2_normalize(normal_vec, axis = 1)


rot_theta = tf.acos(tf.matmul(normalized_normal_vec, tf.transpose(tf.constant([[0.0,0.0,1.0]]),[1,0])))

b_vec = tf.nn.l2_normalize(tf.cross(tf.constant([[0.0,0.0,1.0]]), normalized_normal_vec), axis = 1)

q0 = tf.cos(rot_theta/2.0)
q1 = tf.sin(rot_theta/2.0) * b_vec[0,0]
q2 = tf.sin(rot_theta/2.0) * b_vec[0,1]
q3 = tf.sin(rot_theta/2.0) * b_vec[0,2]

el_0_0 = tf.square(q0) + tf.square(q1) - tf.square(q2) - tf.square(q3)
el_0_1 = 2*(q1*q2-q0*q3)
el_0_2 = 2*(q1*q3+q0*q2)
el_1_0 = 2*(q1*q2+q0*q3)
el_1_1 = tf.square(q0) - tf.square(q1) + tf.square(q2) - tf.square(q3)
el_1_2 = 2*(q2*q3+q0*q1)
el_2_0 = 2*(q1*q3-q0*q2)
el_2_1 = 2*(q2*q3+q0*q1)
el_2_2 = tf.square(q0) - tf.square(q1) - tf.square(q2) + tf.square(q3)

Q = tf.concat([tf.concat([el_0_0,el_0_1,el_0_2], axis = 1), tf.concat([el_1_0,el_1_1,el_1_2], axis = 1), tf.concat([el_2_0,el_2_1,el_2_2], axis = 1)], axis=0)

u_ = tf.matmul(Q,tf.transpose(tf.constant([[1.0,0.0,0.0]]), [1,0]))
v_ = tf.matmul(Q,tf.transpose(tf.constant([[0.0,1.0,0.0]]), [1,0]))
w_ = tf.matmul(Q,tf.transpose(tf.constant([[0.0,0.0,1.0]]), [1,0]))

transform_mat = tf.concat([u_,v_,w_], axis = 1)
    
transformed_coordinates = tf.matmul(centered_points,transform_mat)  

mask = tf.greater(transformed_coordinates[:,2],0)

points_from_side_one = tf.boolean_mask(transformed_coordinates, mask) 

mask2 = tf.less(transformed_coordinates[:,2],0)

points_from_side_two = tf.boolean_mask(transformed_coordinates, mask2) 


indices_one_x = tf.nn.top_k(points_from_side_one[:,0], k=tf.shape(points_from_side_one)[0]).indices
reordered_points_one_x = tf.gather(points_from_side_one, indices_one_x, axis=0)

indices_two_x = tf.nn.top_k(points_from_side_two[:, 0], k=tf.shape(points_from_side_two)[0]).indices
reordered_points_two_x = tf.gather(points_from_side_two, indices_two_x, axis=0)


indices_one_y = tf.nn.top_k(points_from_side_one[:,1], k=tf.shape(points_from_side_one)[0]).indices
reordered_points_one_y = tf.gather(points_from_side_one, indices_one_y, axis=0)

indices_two_y = tf.nn.top_k(points_from_side_two[:, 1], k=tf.shape(points_from_side_two)[0]).indices
reordered_points_two_y = tf.gather(points_from_side_two, indices_two_y, axis=0)





input1_1_x = tf.expand_dims([reordered_points_one_x[:,2]],2)


filter1_1_x = tf.get_variable("v_1", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0))#, trainable=True)

output1_1_x = tf.nn.conv1d(input1_1_x, filter1_1_x, stride=2, padding="SAME")

filter2_1_x = tf.get_variable("v_2", [3, 10, 20], initializer=tf.random_normal_initializer(seed=0))#,trainable=True)

output2_1_x_temp = tf.nn.conv1d(output1_1_x, filter2_1_x, stride=2, padding="SAME")

output2_1_x = tf.cond(tf.shape(output2_1_x_temp)[1] >= 200, lambda: tf.slice(output2_1_x_temp, [0,0,0], [-1,200,-1]), lambda: tf.concat([output2_1_x_temp, tf.zeros([1,200-tf.shape(output2_1_x_temp)[1],20])], axis = 1))


input1_2_x = tf.expand_dims([reordered_points_two_x[:,2]],2)


filter1_2_x = tf.get_variable("v_3", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0))#,trainable=False)

output1_2_x = tf.nn.conv1d(input1_2_x, filter1_2_x, stride=2, padding="SAME")

filter2_2_x = tf.get_variable("v_4", [3, 10, 20], initializer=tf.random_normal_initializer(seed=0))#,trainable=False)

output2_2_x_temp= tf.nn.conv1d(output1_2_x, filter2_2_x, stride=2, padding="SAME")

output2_2_x = tf.cond(tf.shape(output2_2_x_temp)[1] >= 200, lambda: tf.slice(output2_2_x_temp, [0,0,0], [-1,200,-1]), lambda: tf.concat([output2_2_x_temp, tf.zeros([1,200-tf.shape(output2_2_x_temp)[1],20])], axis = 1))



input1_1_y = tf.expand_dims([reordered_points_one_y[:,2]],2)


filter1_1_y = tf.get_variable("v_5", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0))#,trainable=False)

output1_1_y = tf.nn.conv1d(input1_1_y, filter1_1_y, stride=2, padding="SAME")

filter2_1_y = tf.get_variable("v_6", [3, 10, 20],initializer=tf.random_normal_initializer(seed=0))#,trainable=False)

output2_1_y_temp = tf.nn.conv1d(output1_1_y, filter2_1_y, stride=2, padding="SAME")

output2_1_y = tf.cond(tf.shape(output2_1_y_temp)[1] >= 200, lambda: tf.slice(output2_1_y_temp, [0,0,0], [-1,200,-1]), lambda: tf.concat([output2_1_y_temp, tf.zeros([1,200-tf.shape(output2_1_y_temp)[1],20])], axis = 1))




input1_2_y = tf.expand_dims([reordered_points_two_y[:,2]],2)


filter1_2_y = tf.get_variable("v_7", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0))#,trainable=False)

output1_2_y = tf.nn.conv1d(input1_2_y, filter1_2_y, stride=2, padding="VALID")

filter2_2_y = tf.get_variable("v_8", [3, 10, 20], initializer=tf.random_normal_initializer(seed=0))#,trainable=False)

output2_2_y_temp = tf.nn.conv1d(output1_2_y, filter2_2_y, stride=2, padding="VALID")

output2_2_y = tf.cond(tf.shape(output2_2_y_temp)[1] >= 200, lambda: tf.slice(output2_2_y_temp, [0,0,0], [-1,200,-1]), lambda: tf.concat([output2_2_y_temp, tf.zeros([1,200-tf.shape(output2_2_y_temp)[1],20])], axis = 1))


#side_1_descriptor = tf.matmul(tf.transpose(output2_1_x, [0,2,1]), output2_1_y)
#side_2_descriptor = tf.matmul(tf.transpose(output2_2_x, [0,2,1]), output2_2_y)

#print(side_1_descriptor)

# concat_layer = tf.reshape(tf.concat([output2_1_x, output2_2_x, output2_1_y, output2_2_y], axis = 1), [1, 1600])
concat_layer = tf.reshape(tf.concat([output2_1_x ,output2_2_x,output2_1_y,output2_2_y ], axis = 0), [1, 16000])


rot_angles = tf.layers.dense(concat_layer,27)#, trainable=False)
#rot_angles_ = tf.reshape(rot_angles, [3,3,3])

# rotation_matrix_one = tf.squeeze(tf.slice(rot_angles_, [0,0,0], [1,-1,-1]),squeeze_dims=[0])
# rotation_matrix_two =  tf.squeeze(tf.slice(rot_angles_, [1,0,0], [1,-1,-1]),squeeze_dims=[0])
# rotation_matrix_three =  tf.squeeze(tf.slice(rot_angles_, [2,0,0], [1,-1,-1]),squeeze_dims=[0])

rotation_matrix_one = tf.concat([tf.constant([[1.0, 0.0, 0.0]]), [[0.0, tf.cos(rot_angles[0,0]), -tf.sin(rot_angles[0,0])]], [[0.0, tf.cos(rot_angles[0,0]), tf.sin(rot_angles[0,0])]]], axis = 0)
rotation_matrix_two = tf.concat([[[tf.cos(rot_angles[0,1]), 0.0, tf.sin(rot_angles[0,1])]], [[0.0, 1.0, 0.0]], [[-tf.sin(rot_angles[0,1]), 0.0,tf.cos(rot_angles[0,1]) ]]], axis = 0)
rotation_matrix_three = tf.concat([[[tf.cos(rot_angles[0,2]), -tf.sin(rot_angles[0,2]),0.0 ]], [[tf.sin(rot_angles[0,2]), tf.cos(rot_angles[0,2]), 0.0]], [[0.0, 0.0,1.0 ]]], axis = 0)

print(rotation_matrix_one)

centered_points_expanded_ = tf.reshape(transformed_coordinates, [-1, 3])
point_count = tf.shape(centered_points_expanded_)[0]

#rotation_matrix_one = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_one")
trasformed_points_one = tf.matmul(centered_points_expanded_, rotation_matrix_one, name="trans_point_one")
trasformed_points_one_reshaped = tf.reshape(trasformed_points_one, [-1, point_count, 3], name = "trans_point_one_reshape")

#rotation_matrix_two = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_two")
trasformed_points_two = tf.matmul(centered_points_expanded_, rotation_matrix_two, name="trans_point_two")
trasformed_points_two_reshaped = tf.reshape(trasformed_points_two, [-1, point_count, 3], name = "trans_point_two_reshape")

#rotation_matrix_three = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_three")
trasformed_points_three = tf.matmul(centered_points_expanded_, rotation_matrix_three, name="trans_point_three")
trasformed_points_three_reshaped = tf.reshape(trasformed_points_three, [-1, point_count, 3], name = "trans_point_three_reshape")

########################################################################################

trasformed_points_one_reshaped_ = tf.reshape(trasformed_points_one_reshaped, [-1, 3])

#point_distance_one = tf.reduce_sum(tf.square(trasformed_points_one_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

#scale_metric_one = tf.exp(-point_distance_one*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

#scale_metric_tiled_one = tf.tile(scale_metric_one, [1, 3], name="cn_W_tiled")

calibrated_points_one = trasformed_points_one_reshaped_ 



trasformed_points_two_reshaped_ = tf.reshape(trasformed_points_two_reshaped, [-1, 3])

#point_distance_two = tf.reduce_sum(tf.square(trasformed_points_two_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

#scale_metric_two = tf.exp(-point_distance_two*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

#scale_metric_tiled_two = tf.tile(scale_metric_two, [1, 3], name="cn_W_tiled")

calibrated_points_two = trasformed_points_two_reshaped_


trasformed_points_three_reshaped_ = tf.reshape(trasformed_points_three_reshaped, [-1, 3])

#point_distance_three = tf.reduce_sum(tf.square(trasformed_points_three_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

#scale_metric_three = tf.exp(-point_distance_three*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

#scale_metric_tiled_three = tf.tile(scale_metric_three, [1, 3], name="cn_W_tiled")

calibrated_points_three = trasformed_points_three_reshaped_

calibrated_points_one_corrected_shape = tf.reshape(calibrated_points_one, [-1, point_count, 3])


centered_calib_points_one = tf.subtract(calibrated_points_one,tf.reduce_mean(calibrated_points_one,axis=0,keep_dims=True))
centered_calib_points_two = tf.subtract(calibrated_points_two,tf.reduce_mean(calibrated_points_two,axis=0,keep_dims=True))
centered_calib_points_three = tf.subtract(calibrated_points_three,tf.reduce_mean(calibrated_points_three,axis=0,keep_dims=True))

def atan2(y, x):
    angle = tf.select(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle


r_one_temp = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one), axis=1, keepdims = True))
r_one = tf.divide(r_one_temp,tf.reduce_max(r_one_temp, axis = 0, keep_dims = True))
theta_one = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(calibrated_points_one[:,2],1), tf.maximum(r_one_temp,0.001)),0.99),-0.99))
phi_one = tf.atan2(tf.expand_dims(calibrated_points_one[:,1],1),tf.expand_dims(calibrated_points_one[:,0],1))


r_two_temp = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
r_two = tf.divide(r_two_temp,tf.reduce_max(r_two_temp, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_two = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(calibrated_points_two[:,2],1), tf.maximum(r_two_temp,0.001)),0.99),-0.99))
phi_two = tf.atan2(tf.expand_dims(calibrated_points_two[:,1],1),tf.expand_dims(calibrated_points_two[:,0],1))

r_three_temp = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
r_three = tf.divide(r_three_temp,tf.reduce_max(r_three_temp, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_three = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(calibrated_points_three [:,2],1), tf.maximum(r_three_temp,0.001)),0.99),-0.99))
phi_three  = tf.atan2(tf.expand_dims(calibrated_points_three [:,1],1),tf.expand_dims(calibrated_points_three [:,0],1))


########################################################################################

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

r_modified = tf.cond(tf.shape(r)[1] >= 200, lambda: tf.slice(r, [0,0,0], [-1,200,-1]), lambda: tf.concat([r, tf.zeros([3,200-tf.shape(r)[1],1])], axis = 1))


U = tf.concat([u_0_0,u_0_1,u_1_1,u_0_2,u_1_2, u_2_2, u_0_3,u_1_3,u_2_3,u_3_3] ,  axis=2)
V = tf.concat([v_0_0,v_0_1,v_1_1,v_0_2,v_1_2, v_2_2, v_0_3,v_1_3,v_2_3,v_3_3] ,  axis=2)

X_ = tf.concat([U,V] ,  axis=2)
X_modified = tf.cond(tf.shape(X_)[1] >= 200, lambda: tf.slice(X_, [0,0,0], [-1,200,-1]), lambda: tf.concat([X_, tf.zeros([3,200-tf.shape(X_)[1],20])], axis = 1))


#X = tf.matmul(tf.transpose(X_,[0,2,1]), X_)

#mumprint(X_)


#s1, u1, v1 = tf.svd(tf.slice(X_modified, [0,0,0],[-1,20,-1]), full_matrices = True)
#s2, u2, v2 = tf.svd(tf.slice(X_modified, [0,20,0],[-1,20,-1]), full_matrices = True)
#s3, u3, v3 = tf.svd(tf.slice(X_modified, [0,40,0],[-1,20,-1]), full_matrices = True)
#s4, u4, v4 = tf.svd(tf.slice(X_modified, [0,60,0],[-1,20,-1]), full_matrices = True)
#s5, u5, v5 = tf.svd(tf.slice(X_modified, [0,80,0],[-1,20,-1]), full_matrices = True)
#s6, u6, v6 = tf.svd(tf.slice(X_modified, [0,100,0],[-1,20,-1]), full_matrices = True)
#s7, u7, v7 = tf.svd(tf.slice(X_modified, [0,120,0],[-1,20,-1]), full_matrices = True)
#s8, u8, v8 = tf.svd(tf.slice(X_modified, [0,140,0],[-1,20,-1]), full_matrices = True)
#s9, u9, v9 = tf.svd(tf.slice(X_modified, [0,160,0],[-1,20,-1]), full_matrices = True)
#s10, u10, v10 = tf.svd(tf.slice(X_modified, [0,180,0],[-1,20,-1]), full_matrices = True)


#r_temp = tf.slice(r, [0,0,0], [1,-1,-1])

#u = tf.expand_dims(u, 1, name="cn_caps1_output_expanded")

#s1 = tf.expand_dims(s1,2)
#s2 = tf.expand_dims(s2,2)
#s3 = tf.expand_dims(s3,2)
#s4 = tf.expand_dims(s4,2)
#s5 = tf.expand_dims(s5,2)
#s6 = tf.expand_dims(s6,2)
#s7 = tf.expand_dims(s7,2)
#s8 = tf.expand_dims(s8,2)
#s9 = tf.expand_dims(s9,2)
#s10 = tf.expand_dims(s10,2)

#print(s1)
#print(u1)
#print(v1)
#print(r)\

x_rehape = tf.reshape(X_modified,[3,10,20,20])

alpha = 1.0/tf.maximum(tf.reduce_sum(tf.square(x_rehape),[2,3]),0.0001)

alpha_tiled = tf.tile(tf.reshape(alpha,[3,10,1,1]), [1,1,20,20])
#X_0  = tf.multiply(alpha_tiled,tf.transpose(x_rehape,[0,1,3,2] ))

X_0  = 0.01 * tf.transpose(x_rehape,[0,1,3,2] )

e_val,ev = tf.self_adjoint_eig(tf.matmul(tf.transpose(x_rehape,[0,1,3,2]),x_rehape), name="eigendata")


#def body(i):
#	x  = tf.matmul(x, 2.0 *tf.eye(20) - tf.matmul(x_rehape,x))   
 
#	return [tf.add(i,1)]

#def condition_1(i):
#	return tf.less(i,20)

#X_cal, p = tf.while_loop(lambda x, i: i < 70,
#    lambda x, i: ( tf.matmul(x, 2.0 *tf.eye(20) - tf.matmul(x_rehape,x)), i+1),
#    (X_0, 0))


X_cal, p = tf.while_loop(lambda x, i: i < 10,
    lambda x, i:( tf.matmul( tf.eye(20) + 1.0/4.0 * tf.matmul(tf.eye(20)-tf.matmul(x,x_rehape),tf.matmul(3.0*tf.eye(20)-tf.matmul(x,x_rehape),3.0*tf.eye(20)-tf.matmul(x,x_rehape))),x),i+1)
    ,(X_0, 0))

#X_cal, p = tf.while_loop(condition_1,body,[0])
e_1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.eye(20) - tf.matmul(x_rehape,X_cal)),axis=[2,3])))

#X_1 = tf.matmul(X_0, 2.0 *tf.eye(20) - tf.matmul(x_rehape,X_0))
#e_1 = tf.sqrt(tf.reduce_sum(tf.square(tf.eye(20) - tf.matmul(x_rehape,X_1))))

#X_2 = tf.matmul(X_1, 2.0 *tf.eye(20) - tf.matmul(x_rehape,X_1))
#e_2 = tf.sqrt(tf.reduce_sum(tf.square(tf.eye(20) - tf.matmul(x_rehape,X_2))))

#X_3 = tf.matmul(X_2, 2.0 *tf.eye(20) - tf.matmul(x_rehape,X_2))
#e_3 = tf.sqrt(tf.reduce_sum(tf.square(tf.eye(20) - tf.matmul(x_rehape,X_3))))

#X_4 = tf.matmul(X_3, 2.0 *tf.eye(20) - tf.matmul(x_rehape,X_3))
#e_4 = tf.sqrt(tf.reduce_sum(tf.square(tf.eye(20) - tf.matmul(x_rehape,X_1))))

#X_5 = tf.matmul(X_4, 2.0 *tf.eye(20) - tf.matmul(x_rehape,X_4))
#e_5 = tf.sqrt(tf.reduce_sum(tf.square(tf.eye(20) - tf.matmul(x_rehape,X_5))))





B_1 = tf.matmul(tf.matrix_inverse(tf.slice(X_modified, [0,0,0],[-1,20,-1])),tf.slice(r_modified,[0,0,0],[-1,20,-1]))
B_2 = tf.matmul(tf.matrix_inverse(tf.slice(X_modified, [0,20,0],[-1,20,-1])),tf.slice(r_modified,[0,20,0],[-1,20,-1]))
B_3 = tf.matmul(tf.matrix_inverse(tf.slice(X_modified, [0,40,0],[-1,20,-1])),tf.slice(r_modified,[0,40,0],[-1,20,-1]))
B_4 = tf.matmul(tf.matrix_inverse(tf.slice(X_modified, [0,60,0],[-1,20,-1])),tf.slice(r_modified,[0,60,0],[-1,20,-1]))
B_5 = tf.matmul(tf.matrix_inverse(tf.slice(X_modified, [0,80,0],[-1,20,-1])),tf.slice(r_modified,[0,80,0],[-1,20,-1]))
B_6 = tf.matmul(tf.matrix_inverse(tf.slice(X_modified, [0,100,0],[-1,20,-1])),tf.slice(r_modified,[0,100,0],[-1,20,-1]))
B_7 = tf.matmul(tf.matrix_inverse(tf.slice(X_modified, [0,120,0],[-1,20,-1])),tf.slice(r_modified,[0,120,0],[-1,20,-1]))
B_8 = tf.matmul(tf.matrix_inverse(tf.slice(X_modified, [0,140,0],[-1,20,-1])),tf.slice(r_modified,[0,140,0],[-1,20,-1]))
B_9 = tf.matmul(tf.matrix_inverse(tf.slice(X_modified, [0,160,0],[-1,20,-1])),tf.slice(r_modified,[0,160,0],[-1,20,-1]))
B_10 = tf.matmul(tf.matrix_inverse(tf.slice(X_modified, [0,180,0],[-1,20,-1])),tf.slice(r_modified,[0,180,0],[-1,20,-1]))












#B_2 = tf.matrix_solve_ls(tf.slice(X_modified, [0,20,0],[-1,20,-1]),tf.slice(r_modified,[0,20,0],[-1,20,-1]))
#B_3 = tf.matrix_solve_ls(tf.slice(X_modified, [0,40,0],[-1,20,-1]),tf.slice(r_modified,[0,40,0],[-1,20,-1]))
#B_4 = tf.matrix_solve_ls(tf.slice(X_modified, [0,60,0],[-1,20,-1]),tf.slice(r_modified,[0,60,0],[-1,20,-1]))
#B_5 = tf.matrix_solve_ls(tf.slice(X_modified, [0,80,0],[-1,20,-1]),tf.slice(r_modified,[0,80,0],[-1,20,-1]))
#B_6 = tf.matrix_solve_ls(tf.slice(X_modified, [0,100,0],[-1,20,-1]),tf.slice(r_modified,[0,100,0],[-1,20,-1]))
#B_7 = tf.matrix_solve_ls(tf.slice(X_modified, [0,120,0],[-1,20,-1]),tf.slice(r_modified,[0,120,0],[-1,20,-1]))
#B_8 = tf.matrix_solve_ls(tf.slice(X_modified, [0,140,0],[-1,20,-1]),tf.slice(r_modified,[0,140,0],[-1,20,-1]))
#B_9 = tf.matrix_solve_ls(tf.slice(X_modified, [0,160,0],[-1,20,-1]),tf.slice(r_modified,[0,160,0],[-1,20,-1]))
#B_10 = tf.matrix_solve_ls(tf.slice(X_modified, [0,180,0],[-1,20,-1]),tf.slice(r_modified,[0,180,0],[-1,20,-1]))



r_modified_reshape = tf.reshape(r_modified,[3,10,20,1])

B = tf.matmul(X_cal,r_modified_reshape)





#B_1 = tf.matmul(v1,tf.divide(tf.slice(tf.matmul(tf.transpose(u1, perm=[0, 2, 1]),tf.slice(r_modified,[0,0,0],[-1,20,-1])), [0,0,0], [-1,20,-1]),tf.maximum(s1,[[[0.001]]])))
#B_2 = tf.matmul(v2,tf.divide(tf.slice(tf.matmul(tf.transpose(u2, perm=[0, 2, 1]),tf.slice(r_modified,[0,20,0],[-1,20,-1])), [0,0,0], [-1,20,-1]),tf.maximum(s2,[[[0.001]]])))
#B_3 = tf.matmul(v3,tf.divide(tf.slice(tf.matmul(tf.transpose(u3, perm=[0, 2, 1]),tf.slice(r_modified,[0,40,0],[-1,20,-1])), [0,0,0], [-1,20,-1]),tf.maximum(s3,[[[0.001]]])))
#B_4 = tf.matmul(v4,tf.divide(tf.slice(tf.matmul(tf.transpose(u4, perm=[0, 2, 1]),tf.slice(r_modified,[0,60,0],[-1,20,-1])), [0,0,0], [-1,20,-1]),tf.maximum(s4,[[[0.001]]])))
#B_5 = tf.matmul(v5,tf.divide(tf.slice(tf.matmul(tf.transpose(u5, perm=[0, 2, 1]),tf.slice(r_modified,[0,80,0],[-1,20,-1])), [0,0,0], [-1,20,-1]),tf.maximum(s5,[[[0.001]]])))
#B_6 = tf.matmul(v6,tf.divide(tf.slice(tf.matmul(tf.transpose(u6, perm=[0, 2, 1]),tf.slice(r_modified,[0,100,0],[-1,20,-1])), [0,0,0], [-1,20,-1]),tf.maximum(s6,[[[0.001]]])))
#B_7 = tf.matmul(v7,tf.divide(tf.slice(tf.matmul(tf.transpose(u7, perm=[0, 2, 1]),tf.slice(r_modified,[0,120,0],[-1,20,-1])), [0,0,0], [-1,20,-1]),tf.maximum(s7,[[[0.001]]])))
#B_8 = tf.matmul(v8,tf.divide(tf.slice(tf.matmul(tf.transpose(u8, perm=[0, 2, 1]),tf.slice(r_modified,[0,140,0],[-1,20,-1])), [0,0,0], [-1,20,-1]),tf.maximum(s8,[[[0.001]]])))
#B_9 = tf.matmul(v9,tf.divide(tf.slice(tf.matmul(tf.transpose(u9, perm=[0, 2, 1]),tf.slice(r_modified,[0,160,0],[-1,20,-1])), [0,0,0], [-1,20,-1]),tf.maximum(s9,[[[0.001]]])))
#B_10 = tf.matmul(v10,tf.divide(tf.slice(tf.matmul(tf.transpose(u10, perm=[0, 2, 1]),tf.slice(r_modified,[0,180,0],[-1,20,-1])), [0,0,0], [-1,20,-1]),tf.maximum(s10,[[[0.001]]])))

# = tf.concat([B_1, B_2, B_3, B_4, B_5, B_6, B_7, B_8, B_9, B_10], axis = 1)

# estimate = tf.matmul(X,B)
# print(B)
#estimate_1 = tf.matmul(tf.transpose(u, perm=[0, 2, 1]),r_temp)
init_sigma = 0.01

# A_init = tf.random_normal(
# 		  shape=(3, 10,1),
# 		  stddev=init_sigma, dtype=tf.float32, name="cn_W_init")
# A = tf.Variable(A_init, name="cn_W")




# B_init = tf.random_normal(
# 		  shape=(3, 10,1),
# 		  stddev=init_sigma, dtype=tf.float32, name="cn_W_init")
# B = tf.Variable(B_init, name="cn_W")

# estimate = tf.matmul(U,A) + tf.matmul(V,B)

# print(B)

#loss_estimate = tf.losses.absolute_difference(r, estimate)

#optimizer = tf.train.AdamOptimizer()
# grads = optimizer.compute_gradients(loss_estimate)
# train = optimizer.apply_gradients(grads)


caps1_raw = tf.reshape(B, [-1, 60, 10],
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

caps2_n_caps = 2
caps2_n_dims = 16


W_init = tf.random_normal(
    shape=(1, 60, 2, 16, 10),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

batch_size = 1
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

raw_weights = tf.zeros([batch_size, 60, caps2_n_caps, 1, 1],
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
    caps2_output_round_1, [1, 60, 1, 1, 1],
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

caps2_output_norm = tf.reshape(safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm"), [-1,2])

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, caps2_n_caps),
                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1,  caps2_n_caps),
                          name="absent_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")


loss = tf.losses.softmax_cross_entropy(T,caps2_output_norm)
#loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
#grads = optimizer.compute_gradients(loss)
training_op = optimizer.minimize(loss, name="training_op")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

print('before_sess')
sess = tf.Session()
print('after_sess')

def condition(x, i, index, axis):
    return tf.logical_and(tf.equal(x[0,i,0], index), tf.equal(x[0,i,2], axis))
  
  
# test = condition(tf.constant([[[10, 20, 30]]]), 0, 20, 30)



# i = tf.constant(0)
# while_condition = lambda i: tf.less(i, tf.shape(r_modified)[1]-20)
# def body(i):
#     x_concat = tf.slice(r_temp,[0,i,0],[-1,20,-1])
#     # increment i
#     return [tf.add(i, 20)]

# # do the loop:
# r = tf.while_loop(while_condition, body, [i])



  

# block_hankel = tf.slice(calibrated_points_one_corrected_shape, [0, 0, 0], [-1,10,-1])
sess.run(tf.global_variables_initializer())



def read_datapoint(data, filename):
  points = np.array([[0, 0, 0]])
  
  for line in data:
    if 'OFF' != line.strip() and len([float(s) for s in line.strip().split(' ')]) == 3: 
        if 'bathtub' in filename:
           y = [3]
        else:
           y=[1]
        points_list = [float(s) for s in line.strip().split(' ')]
        points = np.append(points,np.expand_dims(np.array(points_list), axis=0), axis = 0)
    
  return points, y

saver = tf.train.Saver()
#saver.restore(sess, "./model.ckpt")
for filename in glob.glob(os.path.join('/home/ram095/sameera/3d_obj/code/3D_object_recognition/test/train/', '*.off')):
	f = open(filename, 'r')
	print(filename)
	points, y_annot = read_datapoint(f, filename)
	points_ = sess.run(e_1, feed_dict = {y:y_annot, raw_points_init:points})
	print(points_)

#	input("Press Enter to continue...")
  #print(t)
	_, loss_train = sess.run([training_op, loss], feed_dict = {y:y_annot, raw_points_init:points})
	print(loss_train)
  
  
saver.save(sess, "./model.ckpt")

# for itr in xrange(100000000):
#             # train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            
            
#             #sess.run(train,feed_dict = {rotation_matrix_one:[[1, 0, 0], [0, 1.0/math.sqrt(2), -1.0/math.sqrt(2)], [0, 1.0/math.sqrt(2), 1.0/math.sqrt(2)]], rotation_matrix_two:[[1, 0, 0], [0, 1.0/2, -math.sqrt(3)/2], [0, math.sqrt(3)/2, 1.0/2]], rotation_matrix_three:[[1, 0, 0], [0, math.sqrt(3)/2, -1.0/2], [0, 1.0/2, math.sqrt(3)/2]], raw_points_init:points})

#             if itr % 1000 == 0:
#                 train_loss = sess.run([loss_estimate], feed_dict = {rotation_matrix_one:[[1, 0, 0], [0, 1.0/math.sqrt(2), -1.0/math.sqrt(2)], [0, 1.0/math.sqrt(2), 1.0/math.sqrt(2)]], rotation_matrix_two:[[1, 0, 0], [0, 1.0/2, -math.sqrt(3)/2], [0, math.sqrt(3)/2, 1.0/2]], rotation_matrix_three:[[1, 0, 0], [0, math.sqrt(3)/2, -1.0/2], [0, 1.0/2, math.sqrt(3)/2]], raw_points_init:points})
#                 print(train_loss)
                

