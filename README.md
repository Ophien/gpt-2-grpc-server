[singularitynet-home]: https://www.singularitynet.io
[author-home]: http://alysson.thegeneralsolution.com

# GPT2 GRPC Server

This repository contains the OpenAi GPT2 transformer neural network GRPC server that can generate text given a seed sentence.

# Requisites

You need to install the following. The requirements.txt file is located in the project's root directory.

```
sudo apt -y install python3-pip
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

# Usage

In order to use this server and client code use the following.

### Server

Run the grpc-server.py in the src directory with

```
python3 grpc-server.py 50021
```

where, it will use the default 117 million parameters model and set the server port to 50021.

### Client

Call the client with

```
python3 grpc-client.py
```

There are three commands available

* generate - Generate text using a specified model and input utterance. Example:

```
python3 grpc-client.py generate --input_sentence "Donald Trump" --model_name 117M --server_address localhost --server_port 50021
```

where, the "Donald Trump" is the seed sentence, 117M is the 117 million parameters model and the others are the server address and port.

* train - train a new model based on a dataset in a txt file. Example:

```
python3 grpc-client.py train --dataset "example_train_datasets/michael_jackson_songs" --iterations 100
```

where, "example_train_datasets/michael_jackson_songs" is the dataset file without extension, and the ***iterations*** parameter is the number of iterations of the training algorithm.

* improvise - generate an improvised sentence based on a trained model. Example:

```
python3 grpc-client.py improvise --model_name michael_jackson_songs
```

where, ***michael_jackson_songs*** is the name of a dataset without its extension and used to previously train the network.

# Docker

There are two docker files, one to run tensorflow in a GPU and the other on a CPU, both are presented next.

## CPU build

To build the CPU docker image run the following command in this project's root directory.

```
docker build -t gpt2-grpc -f Dockerfile.cpu .
```

## GPU build

In contrast, the GPU image can be build with the following command in this project's root directory.

```
docker build -t gpt2-grpc -f Dockerfile.gpu .
```

## Running a container

To run the gpt2-grpc image use the following command.

```
docker run -itd -p 50021:50021 --name gpt2-grpc-container gpt2-grpc python3 grpc-server.py 50021
```

It will run the container, and run the server with the 117 million parameters model in the port 50021.

## Example of generated text

Using the seed Donald Trump, by calling the client, the AI generated and returned the following text.

***Donald Trump's campaign said that, at least temporarily, he would not allow Republicans to join in the fight against Syrian President Bashar al-Assad.***

# Author

[Alysson Ribeiro da Silva][author-home] - *Maintainer* - [SingularityNET][singularitynet-home]