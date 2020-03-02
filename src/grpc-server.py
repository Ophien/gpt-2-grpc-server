from concurrent import futures
from enum import Enum

import grpc

import gpt_2_server_pb2
import gpt_2_server_pb2_grpc

import argparse
import sys
import fire
import json
import os
import numpy as np
import tensorflow as tf
import train
import model
import sample
import encoder
import spacy
import train

class Opperation(Enum):
    GENERATE = 0
    IMPROVISE = 1
    TRAIN = 2
    SINGLE_WORD = 3

class GPT2:
    def __init__(self):
        self.seed = None
        self.nsamples = 1
        self.batch_size = 1
        self.top_k = 40
        self.top_p = 1.0
        self.nlp = spacy.load('en')

    def generate(self, seed_sentence="Universe", model_name='117M', temperature=0.8, opperation=Opperation.IMPROVISE, length=None):
        if self.batch_size is None:
            self.batch_size = 1

        assert self.nsamples % self.batch_size == 0

        self.enc = encoder.get_encoder(model_name)
        self.hparams = model.default_hparams()

        with open(os.path.join('models', model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        if length is None:
            length = self.hparams.n_ctx // 2
        elif length > self.hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % self.hparams.n_ctx)

        ################ SESSION START ################
        # do not use explicit graph with explicit session
        # graph = tf.Graph()
        sess = tf.Session()
        ################ SESSION START ################

        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        ###########################################################################

        def create_generation_tensors():
            context = tf.placeholder(tf.int32, [self.batch_size, None])
            context_tokens = self.enc.encode(seed_sentence)
            output = sample.sample_sequence(
                hparams=self.hparams,
                length=length,
                context=context,
                batch_size=self.batch_size,
                temperature=temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )

            # loading latest model checkpoint
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
            saver.restore(sess, ckpt)

            return [context, context_tokens, output]

        def create_top_k_tensors():
            #return gen_text
            context = tf.placeholder(tf.int32, [self.batch_size, None])
            context_tokens = self.enc.encode(seed_sentence)
            tensor_logits, tensor_indexes = sample.sample_single_word(
                                 hparams=self.hparams,
                                 length=length,
                                 context=context,
                                 batch_size=self.batch_size,
                                 temperature=temperature,
                                 top_k=self.top_k,
                                 top_p=self.top_p
                                 )

            # loading latest model checkpoint
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
            saver.restore(sess, ckpt) 

            return [context, context_tokens, tensor_logits, tensor_indexes]

        def create_improvisation_tensors():
            output = sample.sample_sequence(
                hparams=self.hparams,
                length=length,
                start_token=self.enc.encoder['<|endoftext|>'],
                batch_size=self.batch_size,
                temperature=temperature,
                top_k=self.top_k,
                top_p=self.top_p
                )[:, 1:]

            # loading latest model checkpoint
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
            saver.restore(sess, ckpt)

            return [output]

        ###########################################################################

        def run_generation():
            tensors = create_generation_tensors()
            # avoid to use magic variables
            context = tensors[0]
            context_tokens = tensors[1]
            output = tensors[2]

            # run graph
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(self.batch_size)]
            })[:, len(context_tokens):]

            # decode output into text
            text = self.enc.decode(out[0])

            return text

        def run_top_k():
            tensors = create_top_k_tensors()

            # avoid to use magic variables
            context = tensors[0]
            context_tokens = tensors[1]
            tensor_logits = tensors[2]
            tensor_indexes = tensors[3]

            # run model for logits and index tensors
            logits, indexes = sess.run([tensor_logits, tensor_indexes], feed_dict={context: [context_tokens for _ in range(self.batch_size)]})
            logits_indexes = np.column_stack((logits[0], indexes[0]))

            # sort output according to logits (probabilities)
            def get_prob(logit):
                return logit[0]

            sorted_logits_indexes = sorted(logits_indexes, key=get_prob, reverse=True)
            indexes = np.reshape(np.array(sorted_logits_indexes)[:,1:2], (1,self.top_k))[0]

            # decode words (integers to words)
            text = self.enc.decode(indexes)

            return text

        ###########################################################################
        def run_improvisation():
            tensors = create_improvisation_tensors()

            # run model
            out = sess.run(tensors[0])

            # decode words (integer to words)
            text = self.enc.decode(out[0])

            return text

        text = "NOTHING"

        # generate sentence with n-words
        if opperation == Opperation.GENERATE:
            text = run_generation()

        # generate list of probabilities for the next word
        if opperation == Opperation.SINGLE_WORD:
            text = run_top_k()

        # generate random sentence with n-words
        if opperation == Opperation.IMPROVISE:
            text = run_improvisation()

        ################### CLOSE SESSION ##################
        sess.close()
        ################### CLOSE SESSION ##################

        return text


# this is the implementation of the rpc calls for the defined protobuffer file
class Gpt2Servicer(gpt_2_server_pb2_grpc.gpt2Servicer):
    def __init__(self):
        self.network = GPT2()    

    def Generate(self, request, context):
        print(request.input_seed_sentence)
        generated_text = self.network.generate(seed_sentence=request.input_seed_sentence, model_name=request.input_model_name, opperation=Opperation.GENERATE, length=request.input_length)
        return gpt_2_server_pb2.GenMsg(input_seed_sentence=request.input_seed_sentence, input_model_name=request.input_model_name, output_generated_text=generated_text)

    def Train(self, request, context):
        print(request.input_dataset_path)
        train.custom_train(dataset_path = request.input_dataset_path, run_name = request.input_run_name, iterations=request.input_iterations)
        return gpt_2_server_pb2.TrainMsg(input_dataset_path = request.input_dataset_path, input_iterations = request.input_iterations, output_status = "Done")

    def Improvise(self, request, context):
        print(request.input_model_name)
        generated_text = self.network.generate(model_name=request.input_model_name, opperation=Opperation.IMPROVISE, length=request.input_length)
        return gpt_2_server_pb2.GenMsg(input_seed_sentence="None", input_model_name=request.input_model_name, output_generated_text=generated_text)

    def Top_k(self, request, context):
        print(request.input_seed_sentence)
        generated_text = self.network.generate(seed_sentence=request.input_seed_sentence, model_name=request.input_model_name, opperation=Opperation.SINGLE_WORD, length=request.input_length)
        return gpt_2_server_pb2.GenMsg(input_seed_sentence=request.input_seed_sentence, input_model_name=request.input_model_name, output_generated_text=generated_text)

def serve():
    parser = argparse.ArgumentParser()

    parser.add_argument("server_port",
                       type = str,
                       default = "50021",
                       help = "Server port")

    args = parser.parse_args()

    os.system('reset')

    # create server instance
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # add the services for the gpt2 
    gpt_2_server_pb2_grpc.add_gpt2Servicer_to_server(Gpt2Servicer(), server)

    # set server port
    server.add_insecure_port("[::]:" + args.server_port)

    print("Serving at: " + "0.0.0.0 " + args.server_port)

    # start server and await for it to finish
    server.start()
    server.wait_for_termination()

    print("End serving")

if __name__ == '__main__':
    serve()
