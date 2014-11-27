#include "renderer.h"

#include <zmq.h>
#include <unistd.h>

//const char* myep = "epgm://en0;224.0.0.1:5555";
const char* myep = "tcp://127.0.0.1:5555";

void server() {
    void* zmq_context = zmq_ctx_new();
    void* socket = zmq_socket(zmq_context, ZMQ_PUB);
    int r = zmq_bind(socket, myep);
    if(r < 0) {
        printf("r = %d, %s\n", r, zmq_strerror(errno));
    }
    while(1) {
        int r = zmq_send(socket, "HELLO", 5, 0);
        if(r < 0) {
            printf("r = %d, %s\n", r, zmq_strerror(errno));
        } else {
            printf("Sent 1 packet.\n");
        }
        sleep(1);
    }
}

void client() {
    void* zmq_context = zmq_ctx_new();
    void* socket = zmq_socket(zmq_context, ZMQ_SUB);
    int r = zmq_connect(socket, myep);
    if(r < 0) {
        printf("r = %d, %s\n", r, zmq_strerror(errno));
    }
    zmq_setsockopt(socket, ZMQ_SUBSCRIBE, "", 0);
    while(1) {
        char buffer[100] = { '\0' };
        int r = zmq_recv(socket, buffer, 5, 0);
        if(r < 0) {
            printf("r = %d, %s\n", r, zmq_strerror(errno));
        } else {
            printf("Packet: %s\n", buffer);
        }
        sleep(1);
    }
}

int main(int argc, char* argv[]) {
    if(argc == 1) server();
    else client();
}
