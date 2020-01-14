PYTHON = python
PROTOC = grpc_tools.protoc
INCLUDE = ./proto
SRC = ./src

all: gpt-2-server.proto

gpt-2-server.proto:
	$(PYTHON) -m $(PROTOC) -I$(INCLUDE) --python_out=$(SRC) --grpc_python_out=$(SRC) $@

clean:
	rm -rf $(SRC)/gpt_2_server_pb2_grpc.py $(SRC)/gpt_2_server_pb2.py
