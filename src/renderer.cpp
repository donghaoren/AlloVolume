#include "renderer.h"

#include "configparser.h"

#include <zmq.h>
#include <unistd.h>

#include <string>

#define PACKET_SIZE 60000

using namespace std;

ConfigParser config;

struct packet_header_t {
    int sequence_number;
    double time;
};

void server() {
    void* zmq_context = zmq_ctx_new();
    void* socket = zmq_socket(zmq_context, ZMQ_PUB);
    int value;
    value = config.get<int>("zmq.sndhwm", 10000); zmq_setsockopt(socket, ZMQ_SNDHWM, &value, sizeof(int));
    value = config.get<int>("zmq.sndbuf", 0); zmq_setsockopt(socket, ZMQ_SNDBUF, &value, sizeof(int));
    value = config.get<int>("zmq.rate", 10000000); zmq_setsockopt(socket, ZMQ_RATE, &value, sizeof(int));

    int r = zmq_bind(socket, config.get<string>("renderer.broadcast").c_str());
    if(r < 0) {
        printf("r = %d, %s\n", r, zmq_strerror(errno));
    }
    int test_size = config.get<int>("renderer.packet_size");
    char* data = new char[test_size];
    for(int i = 0; i < test_size; i++) {
        data[i] = (char)i;
    }
    while(1) {
        int r = zmq_send(socket, data, test_size, 0);
        if(r < 0) {
            printf("r = %d, %s\n", r, zmq_strerror(errno));
        } else {
            printf("Sent 1 packet.\n");
        }
        usleep(500000);
    }
}

void client() {
    void* zmq_context = zmq_ctx_new();
    void* socket = zmq_socket(zmq_context, ZMQ_SUB);
    int value;
    value = config.get<int>("zmq.rcvhwm", 10000); zmq_setsockopt(socket, ZMQ_RCVHWM, &value, sizeof(int));
    value = config.get<int>("zmq.rcvbuf", 0); zmq_setsockopt(socket, ZMQ_RCVBUF, &value, sizeof(int));
    value = config.get<int>("zmq.rate", 10000000); zmq_setsockopt(socket, ZMQ_RATE, &value, sizeof(int));
    int r = zmq_connect(socket, config.get<string>("renderer.broadcast").c_str());
    if(r < 0) {
        printf("r = %d, %s\n", r, zmq_strerror(errno));
    }
    zmq_setsockopt(socket, ZMQ_SUBSCRIBE, "", 0);
    int test_size = config.get<int>("renderer.packet_size");
    char* data = new char[test_size];
    while(1) {
        int r = zmq_recv(socket, data, test_size, 0);
        if(r < 0) {
            printf("r = %d, %s\n", r, zmq_strerror(errno));
        } else {
            bool failure = false;
            for(int i = 0; i < test_size; i++) {
                if(data[i] != (char)i) failure = true;
            }
            if(failure)
                printf("Packet Failed.\n");
            else
                printf("Packet: %d\n", test_size);
        }
    }
}

int main(int argc, char* argv[]) {
    char hostname[256];
    gethostname(hostname, 256);
    config.parseFile("allovolume.yaml", hostname);
    config.parseFile("allovolume.yaml");
    if(config.get<string>("renderer.role") == "server") server();
    if(config.get<string>("renderer.role") == "client") client();
}
