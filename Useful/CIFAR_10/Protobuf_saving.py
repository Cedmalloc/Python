import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from keras.models import load_model
import keras.backend as K

def freeze(session, keep_var_names = None, output_names = None, clear_devices = True):
    graph = session.graph()  #sessoin graph is like the model
    with graph.as_default(): #get default value
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        #list of all variable names in graph
        output_names = output_names or [] #operations for final frozen graph
        output_names +=[v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def() #definition is like shape of the graph
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''  #set to empty line
        frozen_graph = convert_variables_to_constants(sess = session, input_graph = input_graph_def , output_node_names = output_names,
        variable_names_whitelist = freeze_var_names) #converts variable nodes in to constants

        return frozen_graph

model = load_model(filepath= 'Image-classifier.h5')

K.set_learning_phase(1) #helps eliminate problem

frozen_graph = freeze_graph(K.get_session(),output_names = [model.output.op.name])
tf.train.write(frozen_graph, '-', 'Image-classifier.pb',False) #set it to binary file with False
#write the file
