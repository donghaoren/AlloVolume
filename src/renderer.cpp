#include "renderer.h"

#include <yaml-cpp/yaml.h>
#include <zmq.h>
#include <unistd.h>

#include <string>

using namespace std;

YAML::Node config;

void server() {
    void* zmq_context = zmq_ctx_new();
    void* socket = zmq_socket(zmq_context, ZMQ_PUB);
    int r = zmq_bind(socket, config["renderer"]["broadcast"].as<string>().c_str());
    if(r < 0) {
        printf("r = %d, %s\n", r, zmq_strerror(errno));
    }
    int test_size = 1 * 1000000;
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
        usleep(1000000);
    }
}

void client() {
    void* zmq_context = zmq_ctx_new();
    void* socket = zmq_socket(zmq_context, ZMQ_SUB);
    int r = zmq_connect(socket, config["renderer"]["broadcast"].as<string>().c_str());
    if(r < 0) {
        printf("r = %d, %s\n", r, zmq_strerror(errno));
    }
    zmq_setsockopt(socket, ZMQ_SUBSCRIBE, "", 0);
    int test_size = 100 * 1000000;
    char* data = new char[test_size];
    while(1) {
        char buffer[100] = { '\0' };
        int r = zmq_recv(socket, data, test_size, 0);
        if(r < 0) {
            printf("r = %d, %s\n", r, zmq_strerror(errno));
        } else {
            bool failure = false;
            for(int i = 0; i < test_size; i++) {
                if(data[i] != (char)i) failure = true;
            }
            if(failure)
                printf("Packet Failed: %s\n", buffer);
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
