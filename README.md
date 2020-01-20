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
python3 grpc-server.py "117M" "50021"
```

where, 117M is the 117 million parameters model and 50021 is the server port.

### Client

Call the client with

```
python3 grpc-client.py "Donald Trump" "117M" "localhost" "50021"
```

where, the "Donald Trump" is the seed sentence, 117M is the 117 million parameters model and the others are the server address and port.

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
docker run -itd -p 50021:50021 --name gpt2-grpc-container gpt2-grpc python3 grpc-server.py "117M" "50021"
```

It will run the container, and run the server with the 117 million parameters model in the port 50021.

## Using the client

The client, in the src directory, can be used with the following.

```
python3 grpc-client.py "Donal Trump" "117M" "localhost" "50021"
```

With it, you will call the server, through the port 50021 inside your container, to generate a text with the seed "Donald Trump" using the 117 million parameters model.

## Example of generated text

Using the seed Donald Trump, by calling the client, the AI generated and returned the following text.

***Donald Trump's campaign said that, at least temporarily, he would not allow Republicans to join in the fight against Syrian President Bashar al-Assad.***

# Author

[Alysson Ribeiro da Silva][author-home] - *Maintainer* - [SingularityNET][singularitynet-home]