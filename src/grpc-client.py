from __future__ import print_function
import random
import sys
import grpc
import gpt_2_server_pb2
import gpt_2_server_pb2_grpc
import argparse

def gen(stub, input_sentence, model_name, length):
    input_msg = gpt_2_server_pb2.GenMsg(input_seed_sentence=input_sentence, input_model_name=model_name, input_length=length, output_generated_text="none")
    output_msg = stub.Generate(input_msg).output_generated_text
    return output_msg

def train(stub, dataset_path, iterations):
    input_msg = gpt_2_server_pb2.TrainMsg(input_dataset_path = dataset_path, input_iterations=iterations, output_status = "None")
    output_msg = stub.Train(input_msg).output_status
    return output_msg

def improvise(stub, model_name, length):
    input_msg = gpt_2_server_pb2.GenMsg(input_seed_sentence="None", input_model_name=model_name, input_length=length, output_generated_text="none")
    output_msg = stub.Improvise(input_msg).output_generated_text
    return output_msg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("command",
                       type = str,
                       default = "gen",
                       help = "[generate][train][improvise]")

    parser.add_argument("--input_sentence",
                        type = str,
                        default = "universe",
                        help = "Sentence used to generate text")

    parser.add_argument("--model_name",
                        type = str,
                        default = "universe",
                        help = "Model to be used")

    parser.add_argument("--dataset",
                        type = str,
                        default = "example_train_dataset/michael_jackson_songs",
                        help = "Dataset file path without extension")

    parser.add_argument("--iterations",
                        type = int,
                        default = 10,
                        help = "number of the training iterations")

    parser.add_argument("--length",
                        type = int,
                        default = 100,
                        help = "number of words to be generated")

    parser.add_argument("--server_address",
                        type = str,
                        default = "localhost",
                        help = "Server address")

    parser.add_argument("--server_port",
                        type = str,
                        default = "50021",
                        help = "Server port")

    args = parser.parse_args()

    channel = grpc.insecure_channel(args.server_address + ":" + args.server_port)
    stub = gpt_2_server_pb2_grpc.gpt2Stub(channel)

    if args.command == "generate":
        generated_text = gen(stub, args.input_sentence, args.model_name, args.length)
        print(generated_text)

    if args.command == "train":
        training_output = train(stub, args.dataset, args.iterations)
        print(training_output)

    if args.command == "improvise":
        generated_text = improvise(stub, args.model_name, args.length)
        print(generated_text)