all:
	protoc allovolume_protocol.proto --cpp_out ../src --python_out ../controller/server
	cp ../controller/server/allovolume_protocol_pb2.py ../utilities/allovolume
