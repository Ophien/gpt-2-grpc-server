from __future__ import print_function
import random
import sys
import grpc
import gpt_2_server_pb2
import gpt_2_server_pb2_grpc
import argparse

def gen(stub, input_sentence, model_name):
    input_msg = gpt_2_server_pb2.GenMsg(input_seed_sentence=input_sentence, input_model_name=model_name, output_generated_text="none")
    output_msg = stub.Generate(input_msg).output_generated_text
    return output_msg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input_sentence",
                        type = str,
                        default = "universe",
                        help = "Sentence used to generate text")

    parser.add_argument("model_name",
                        type = str,
                        default = "universe",
                        help = "Model to be used")

    parser.add_argument("server_address",
                        type = str,
                        default = "localhost",
                        help = "Server address")

    parser.add_argument("server_port",
                        type = str,
                        default = "50021",
                        help = "Server port")

    args = parser.parse_args()

    channel = grpc.insecure_channel(args.server_address + ":" + args.server_port)
    stub = gpt_2_server_pb2_grpc.gpt2Stub(channel)

    generated_text = gen(stub, args.input_sentence, args.model_name)

    print(generated_text)
