#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:31:39 2018

@author: satyendra.kumar9175
"""

import tensorflow as tf

#Tensorflow works on graphs where the nodes are the tensors and the edges are the operations on those tensors
#A tensor can be a constant or a variable
#An example of an operation is tf.add() which adds two values

#Creating a graph
g = tf.Graph()

#Establish the graph as the default graph
with g.as_default():
    # Assemble a graph consisting of the following three operations:
    #   * Two tf.constant operations to create the operands.
    #   * One tf.add operation to add the two operands.
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    z = tf.constant(4, name="z_const")
    my_sum = tf.add(x, y, name="x_y_sum")
    final_sum = tf.add(my_sum, z, name="x_y_z_sum")
    
    #Now create a session because a the state of a graph is stored in a session
    with tf.Session() as sess:
        print final_sum.eval()
        

        
