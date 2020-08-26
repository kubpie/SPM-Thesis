#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
#

import time

import tensorflow as tf

from pathlib import Path

#from kglib.kgcn.learn.feed import create_placeholders, create_feed_dict, make_all_runnable_in_session
#from kglib.kgcn.learn.loss import loss_ops_preexisting_no_penalty
from feed_mod import create_placeholders, create_feed_dict, create_batches_from_input, make_all_runnable_in_session
from loss_mod import loss_ops_preexisting_no_penalty, loss_ops_from_difference
from kglib.kgcn.learn.metrics import existence_accuracy
from average_gradients import calc_average_grad
from graph_nets import utils_np
from graph_nets.graphs import GraphsTuple


class KGCNLearner:
    """
    Responsible for running a KGCN model
    """
    def __init__(self, model, save_fle, reload_fle, log_dir, num_processing_steps_tr=10, num_processing_steps_ge=10):
        """Args:
            save_fle: Name to save the trained model to.
            reload_fle: Name to load saved model from, when doing inference.
        """
        self._log_dir = log_dir
        self._save_fle = save_fle
        self._reload_fle = reload_fle
        self._model = model
        self._num_processing_steps_tr = num_processing_steps_tr
        self._num_processing_steps_ge = num_processing_steps_ge

    def train(self,
                 tr_input_graphs,
                 tr_target_graphs,
                 ge_input_graphs,
                 ge_target_graphs,
                 num_training_iterations=1000,
                 learning_rate=1e-3,
                 log_every_epochs=20,
                 clip = 5.0,
                 weighted = False):
        """
        Args:
            tr_graphs: In-memory graphs of Grakn concepts for training
            ge_graphs: In-memory graphs of Grakn concepts for generalisation
            num_processing_steps_tr: Number of processing (message-passing) steps for training.
            num_processing_steps_ge: Number of processing (message-passing) steps for generalization.
            num_training_iterations: Number of training iterations
            log_every_seconds: The time to wait between logging and printing the next set of results.
            log_dir: Directory to store TensorFlow events files, output graphs & saved models!

        Returns:

        """
        save_fle = Path(self._save_fle)
        print(f'Saving output to directory:{save_fle}\n')

        tf.set_random_seed(42)

        #### TODO: SPLIT INPUT GRAPHS INTO MANAGEABLE BATCHES
        # Split input graphs into mini-batches
        #batch_size = 2 # TOTAL number of graphs per batch
        #training_batches = create_batches_from_input(tr_input_graphs, batch_size = batch_size)
        #for tr_input_graphs in training_batches:

        # Create placeholders and define tf training
        input_ph, target_ph = create_placeholders(tr_input_graphs, tr_target_graphs)

        # A list of outputs, one per processing step.
        output_ops_tr = self._model(input_ph, self._num_processing_steps_tr)
        output_ops_ge = self._model(input_ph, self._num_processing_steps_ge)

        # Training loss        
        loss_ops_tr = loss_ops_preexisting_no_penalty(target_ph, output_ops_tr, weighted = weighted) #LOSS FUNCTION
        # Loss across processing steps.
        loss_op_tr = sum(loss_ops_tr) / self._num_processing_steps_tr

        tf.summary.scalar('loss_op_tr', loss_op_tr) #this should add training loss to a filewriter log
        # Test/generalization loss.
        loss_ops_ge = loss_ops_preexisting_no_penalty(target_ph, output_ops_ge, weighted = weighted) #LOSS FUNCTION
        loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.
        tf.summary.scalar('loss_op_ge', loss_op_ge) #this should add generalisation loss to a filewriter log


        # Optimizer
        # TODO: Optimize learning rate?? Adaptive learning_raye\sqrt(time) for example? vars: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss_op_tr))
        
        average_grads = calc_average_grad(gradients) #my add
        print(average_grads)

        for grad, var in zip(gradients, variables):
            try:
                print(var.name)
                tf.summary.histogram('gradients/' + var.name, grad)
            except:
                pass

        gradients, _ = tf.clip_by_global_norm(gradients, clip) #clip = 5.0
        step_op = optimizer.apply_gradients(zip(gradients, variables))

        input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)
        
        # This cell resets the Tensorflow session, but keeps the same computational
        # graph.
        #try:
        #    sess.close()
        #except NameError:
        #    pass

        sess = tf.Session()
        merged_summaries = tf.summary.merge_all()

        #train_writer = None #turned on the train writer! was None

        if self._log_dir is not None:
            print(f'\nFileWriter: {self._log_dir}')
            train_writer = tf.summary.FileWriter(self._log_dir, sess.graph)
            #scalar_writer = tf.summary.FileWriter(self._log_dir+'/scalars/',sess.graph)
            #train_writer = tf.compat.v1.summary.FileWriter(self._log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        model_saver = tf.train.Saver()

        logged_iterations = []
        losses_tr = []
        corrects_tr = []
        solveds_tr = []
        losses_ge = []
        corrects_ge = []
        solveds_ge = []

        print("# (iteration number), T (elapsed seconds), "
              "Ltr (training loss), Lge (test/generalization loss), "
              "Ctr (training fraction nodes/edges labeled correctly), "
              "Str (training fraction examples solved correctly), "
              "Cge (test/generalization fraction nodes/edges labeled correctly), "
              "Sge (test/generalization fraction examples solved correctly)")

        start_time = time.time()
        for iteration in range(num_training_iterations):
            feed_dict = create_feed_dict(input_ph, target_ph, tr_input_graphs, tr_target_graphs)
            if iteration % log_every_epochs == 0:

                train_values = sess.run(
                    {
                        "step": step_op,
                        "target": target_ph,
                        "loss": loss_op_tr,
                        "outputs": output_ops_tr,
                        "summary": merged_summaries
                    },
                    feed_dict=feed_dict)

                if train_writer is not None:
                    #print(f'Added summary to writer')
                    train_writer.add_summary(train_values["summary"], iteration)

                feed_dict = create_feed_dict(input_ph, target_ph, ge_input_graphs, ge_target_graphs)
                test_values = sess.run(
                    {
                        "target": target_ph,
                        "loss": loss_op_ge,
                        "outputs": output_ops_ge
                    },
                    feed_dict=feed_dict)

                #print(f'target: {train_values["target"]}') #my add
                #print(f'output: {train_values["outputs"]}')
                correct_tr, solved_tr = existence_accuracy(
                    train_values["target"], train_values["outputs"][-1], use_edges=False)
                correct_ge, solved_ge = existence_accuracy(
                    test_values["target"], test_values["outputs"][-1], use_edges=False)

                elapsed = time.time() - start_time
                losses_tr.append(train_values["loss"])
                corrects_tr.append(correct_tr)
                solveds_tr.append(solved_tr)
                losses_ge.append(test_values["loss"])
                corrects_ge.append(correct_ge)
                solveds_ge.append(solved_ge)
                logged_iterations.append(iteration)
                print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Ctr {:.4f}, Str"
                      " {:.4f}, Cge {:.4f}, Sge {:.4f}".format(
                        iteration, elapsed, train_values["loss"], test_values["loss"],
                        correct_tr, solved_tr, correct_ge, solved_ge))
            else:
                train_values = sess.run(
                    {
                        "step": step_op,
                        "target": target_ph,
                        "loss": loss_op_tr,
                        "outputs": output_ops_tr
                    },
                    feed_dict=feed_dict)

        # Train the model and save it in the end
        # TODO: Could modify saver to save model checkpoint every n-epochs
        if not save_fle.is_dir():
            model_saver.save(sess, save_fle.as_posix())
            #save_fle.with_suffix('.pbtxt').as_posix() = 
            tf.train.write_graph(sess.graph.as_graph_def(), logdir=self._log_dir, name='graph_model.pbtxt', as_text=True) 
            #print(f'Saved model to {log_dir+save_fle}')
        training_info = logged_iterations, losses_tr, losses_ge, corrects_tr, corrects_ge, solveds_tr, solveds_ge
        return train_values, test_values, training_info#, feed_dict
    
    ###############################
    # VALIDATION WITHOUT TRAINING #
    ###############################
    
    # New function to infer / apply without training
    # Inspired from: https://medium.com/@prasadpal107/saving-freezing-optimizing-for-inference-restoring-of-tensorflow-models-b4146deb21b5
    def infer(self, input_graphs, target_graphs):

        reload_file = Path(self._reload_fle)

        input_ph, target_ph = create_placeholders(input_graphs, target_graphs)
        input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)
        output_ops_ge = self._model(input_ph, self._num_processing_steps_ge)
        saver = tf.train.import_meta_graph(reload_file.as_posix() + '.meta')
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()
        with sess.as_default():
            if not reload_file.is_dir():
                saver.restore(sess, reload_file.as_posix())
            else:
                print("no file found, restoring failed")
            
            input_graphs_tuple = utils_np.networkxs_to_graphs_tuple(input_graphs)
            target_graphs_tuple = utils_np.networkxs_to_graphs_tuple(target_graphs)
            feed_dict = {
                input_ph: input_graphs_tuple,
                target_ph: target_graphs_tuple,
            }
            test_values = sess.run(
                {
                    "target": target_ph,
                    "outputs": output_ops_ge,
                },
                feed_dict=feed_dict)
            
            correct_ge, solved_ge = existence_accuracy(
                test_values["target"], test_values["outputs"][-1], use_edges=False)
            
            testing_info = 0, 0, 0, 0, [correct_ge], 0, [solved_ge]

        return test_values, testing_info