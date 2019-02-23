# -*- coding: utf-8 -*-
"""Tensorflow_basics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LMxUC1iAYvofkJuDT9l4GgQpQOHCFsIv
"""

import tensorflow as tf

#constant nodes and simple operations
const_node1 = tf.constant(1.0, dtype = tf.float32) #access to constant node
const_node2 = tf.constant(2.0,dtype = tf.float32) #dtype of float 32
const_node3 = tf.constant([6.0,7.0,8.0],dtype = tf.float32)

adder_node = tf.add(const_node1, const_node2)
#multiply, divide, subtract works the same 

session = tf.Session()

a = session.run(adder_node) #outputs Tensor(array) of 1 and 2


print(a)

#placeholders

pl_holder1 = tf.placeholder(dtype=tf.float32)
pl_holder2 = tf.placeholder(dtype=tf.float32)
multiply_node = tf.multiply(pl_holder1, pl_holder2)

session = tf.Session() #create session

print(session.run(multiply_node,{pl_holder1:9.0, pl_holder2:8.0} ))   #runs the values specified

#variable nodes
#inital value can be changed

var_node1 = tf.Variable(1.0, dtype = tf.float32) #access to constant node
const_node1 = tf.constant(2.0,dtype = tf.float32) #dtype of float 32


adder_node = tf.add(const_node1, var_node1)
#multiply, divide, subtract works the same 

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
session.run(var_node1.assign(20))
session.run(adder_node) #outputs Tensor(array) of 1 and 2