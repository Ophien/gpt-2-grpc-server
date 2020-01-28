from concurrent import futures

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

class GPT2:
    def __init__(self):
        self.seed = None
        self.nsamples = 1
        self.batch_size = 1
        self.top_k = 40
        self.top_p = 1.0
        self.nlp = spacy.load('en')

    def generate(self, seed_sentence="Universe", model_name='117M', temperature=0.8, improvise=False, length=None):
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

        subject = ""
        indirect_object = ""
        direct_object = ""

        if improvise is False:
            # pre-process input sentence to get subject, direct object, and indirect object
            parsed_text = self.nlp(seed_sentence)

            for text in parsed_text:
                if text.dep_ == "nsubj":
                    subject += text.orth_ + " "
                if text.dep_ == "iobj":
                    indirect_object += text.orth_ + " "
                if text.dep_ == "dobj":
                    direct_object += text.orth_ + " "

        with tf.Session(graph=tf.Graph()) as sess:
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)
            context = None
            output = None

            if improvise is False:
                context = tf.placeholder(tf.int32, [self.batch_size, None])
                output = sample.sample_sequence(
                    hparams=self.hparams,
                    length=length,
                    context=context,
                    batch_size=self.batch_size,
                    temperature=temperature,
                    top_k=self.top_k,
                    top_p=self.top_p
                )
            else:
                output = sample.sample_sequence(
                    hparams=self.hparams, 
                    length=length,
                    start_token=self.enc.encoder['<|endoftext|>'],
                    batch_size=self.batch_size,
                    temperature=temperature, 
                    top_k=self.top_k, 
                    top_p=self.top_p
                )[:, 1:]

            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
            saver.restore(sess, ckpt)

            if improvise is False:
                context_tokens = self.enc.encode(seed_sentence)
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(self.batch_size)]
                })[:, len(context_tokens):]

                text = self.enc.decode(out[0])
            else:
                out = sess.run(output)
                text = self.enc.decode(out[0])

            return text

# this is the implementation of the rpc calls for the defined protobuffer file
class Gpt2Servicer(gpt_2_server_pb2_grpc.gpt2Servicer):
    def __init__(self):
        self.network = GPT2()    

    def Generate(self, request, context):
        print(request.input_seed_sentence)
        generated_text = self.network.generate(seed_sentence=request.input_seed_sentence, model_name=request.input_model_name, improvise=False, length=request.input_length)
        return gpt_2_server_pb2.GenMsg(input_seed_sentence=request.input_seed_sentence, input_model_name=request.input_model_name, output_generated_text=generated_text)

    def Train(self, request, context):
        print(request.input_dataset_path)
        train.custom_train(dataset_path = request.input_dataset_path, iterations=request.input_iterations)
        return gpt_2_server_pb2.TrainMsg(input_dataset_path = request.input_dataset_path, input_iterations = request.input_iterations, output_status = "Done")

    def Improvise(self, request, context):
        print(request.input_model_name)
        generated_text = self.network.generate(model_name=request.input_model_name, improvise=True, length=request.input_length)
        return gpt_2_server_pb2.GenMsg(input_seed_sentence="None", input_model_name=request.input_model_name, output_generated_text=generated_text)

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