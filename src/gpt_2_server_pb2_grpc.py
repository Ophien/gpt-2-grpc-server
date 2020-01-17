# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import gpt_2_server_pb2 as gpt__2__server__pb2


class gpt2Stub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Generate = channel.unary_unary(
        '/gpt2server.gpt2/Generate',
        request_serializer=gpt__2__server__pb2.GenMsg.SerializeToString,
        response_deserializer=gpt__2__server__pb2.GenMsg.FromString,
        )


class gpt2Servicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Generate(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_gpt2Servicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Generate': grpc.unary_unary_rpc_method_handler(
          servicer.Generate,
          request_deserializer=gpt__2__server__pb2.GenMsg.FromString,
          response_serializer=gpt__2__server__pb2.GenMsg.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'gpt2server.gpt2', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))