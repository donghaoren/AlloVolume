#include "renderer.h"

#include <yaml-cpp/yaml.h>
#include <zmq.h>
#include <unistd.h>

#include <string>

#define PACKET_SIZE 60000

using namespace std;

YAML::Node config;

void server() {
    void* zmq_context = zmq_ctx_new();
    void* socket = zmq_socket(zmq_context, ZMQ_PUB);
    int value;
    value = 200 * 1024 * 1024; zmq_setsockopt(socket, ZMQ_SNDHWM, &value, sizeof(int));
    value = 64 * 1024 * 1024; zmq_setsockopt(socket, ZMQ_SNDBUF, &value, sizeof(int));
    value = 1000; zmq_setsockopt(socket, ZMQ_RATE, &value, sizeof(int));

    int r = zmq_bind(socket, config["renderer"]["broadcast"].as<string>().c_str());
    if(r < 0) {
        printf("r = %d, %s\n", r, zmq_strerror(errno));
    }
    int test_size = config["renderer"]["packet_size"].as<int>();
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
    value = 200 * 1024 * 1024; zmq_setsockopt(socket, ZMQ_RCVHWM, &value, sizeof(int));
    value = 64 * 1024 * 1024; zmq_setsockopt(socket, ZMQ_RCVBUF, &value, sizeof(int));
    value = 1000; zmq_setsockopt(socket, ZMQ_RATE, &value, sizeof(int));
    int r = zmq_connect(socket, config["renderer"]["broadcast"].as<string>().c_str());
    if(r < 0) {
        printf("r = %d, %s\n", r, zmq_strerror(errno));
    }
    zmq_setsockopt(socket, ZMQ_SUBSCRIBE, "", 0);
    int test_size = config["renderer"]["packet_size"].as<int>();
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
        sleep(1);
    }
}

int main(int argc, char* argv[]) {
    config = YAML::LoadFile("allovolume.yaml");
    if(argc == 2) client();
    else server();
}
