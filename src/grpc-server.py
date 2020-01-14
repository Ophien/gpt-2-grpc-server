from concurrent import futures

import grpc

import gpt_2_server_pb2
import gpt_2_server_pb2_grpc

# this is the implementation of the rpc calls for the defined protobuffer file
class Gpt2Servicer(gpt_2_server_pb2_grpc.gpt2Servicer):
    def __init__(self):
        self.data = "none"

def serve():
    # create server instance
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

    # add the services for the gpt2 
    gpt_2_server_pb2_grpc.add_gpt2Servicer_to_server(Gpt2Servicer(), server)

    # set server port
    server.add_insecure_port('[::]:50021')
    
    # start server and await for it to finish
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    print("Serving at: " + "0.0.0.0 " + "50021 ")
    serve()
