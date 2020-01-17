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

class GPT2:
    def __init__(self, 
                 model_name = '117M', 
                 seed = None, 
                 nsamples = 1,
                 batch_size = 1,
                 length = None,
                 temperature = 0.8,
                 top_k = 40,
                 top_p = 1.0):

        self.model_name = model_name
        self.seed = seed
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.length = length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def configure(self):
        if self.batch_size is None:
            self.batch_size = 1

        assert self.nsamples % self.batch_size == 0

        self.enc = encoder.get_encoder(self.model_name)
        self.hparams = model.default_hparams()

        with open(os.path.join('models', self.model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        if self.length is None:
            self.length = self.hparams.n_ctx // 2
        elif self.length > self.hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % self.hparams.n_ctx)

    def generate(self, seed_sentence):
        with tf.Session(graph=tf.Graph()) as sess:
            context = tf.placeholder(tf.int32, [self.batch_size, None])
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)
            
            output = sample.sample_sequence(
                hparams=self.hparams,
                length=self.length,
                context=context,
                batch_size=self.batch_size,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )
            
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join('models', self.model_name))
            saver.restore(sess, ckpt)
            raw_text = seed_sentence
            
            context_tokens = self.enc.encode(raw_text)
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(self.batch_size)]
            })[:, len(context_tokens):]

            text = self.enc.decode(out[0])
            output_text = text.split('.')[0] + (".")

            return output_text

# this is the implementation of the rpc calls for the defined protobuffer file
class Gpt2Servicer(gpt_2_server_pb2_grpc.gpt2Servicer):
    def __init__(self,
                 model_name = '117M', 
                 seed = None, 
                 nsamples = 1,
                 batch_size = 1,
                 length = None,
                 temperature = 0.8,
                 top_k = 40,
                 top_p = 1.0):

        self.network = GPT2(model_name = model_name,
                            seed = seed,
                            nsamples = nsamples,
                            batch_size = batch_size,
                            length = length,
                            temperature = temperature,
                            top_k = top_k,
                            top_p = top_p)

        self.network.configure()        

    def Generate(self, request, context):
        print(request.input_seed_sentence)
        generated_text = self.network.generate(request.input_seed_sentence)
        return gpt_2_server_pb2.GenMsg(input_seed_sentence="none", input_model_name="117M", output_generated_text=generated_text)

def serve():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name",
                        type = str,
                        default = "universe",
                        help = "Model to be loaded")

    parser.add_argument("server_port",
                       type = str,
                       default = "50021",
                       help = "Server port")

    args = parser.parse_args()

    os.system('reset')

    print("Loaded model: " + args.model_name)

    # create server instance
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # add the services for the gpt2 
    gpt_2_server_pb2_grpc.add_gpt2Servicer_to_server(Gpt2Servicer(model_name = args.model_name), server)

    # set server port
    server.add_insecure_port("[::]:" + args.server_port)

    print("Serving at: " + "0.0.0.0 " + args.server_port)

    # start server and await for it to finish
    server.start()
    server.wait_for_termination()

    print("End serving")

if __name__ == '__main__':
    serve()
