from __future__ import print_function

import os
import sys

import tensorflow.compat.v1 as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file

tf.disable_v2_behavior()

if __name__ == "__main__":
    cfg_from_file('ctpn/text.yml')

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    net = get_network("VGGnet_test")
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()
    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
    print(' done.')

    print('all nodes are:\n')
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    node_names = [node.name for node in input_graph_def.node]
    for x in node_names:
        print(x)
    output_node_names = 'Reshape_2,rpn_bbox_pred/Reshape_1'
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(',')
    )
    output_graph = 'data/ctpn.pb'
    with tf.gfile.GFile(output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()
