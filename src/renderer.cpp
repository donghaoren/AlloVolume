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


#ifndef _WIN32
#include <sys/time.h>
double getPreciseTime() {
    timeval t;
    gettimeofday(&t, 0);
    double s = t.tv_sec;
    s += t.tv_usec / 1000000.0;
    return s;
}
#else
#include <windows.h>
double getPreciseTime() {
    LARGE_INTEGER data, frequency;
    QueryPerformanceCounter(&data);
    QueryPerformanceFrequency(&frequency);
    return (double)data.QuadPart / (double)frequency.QuadPart;
    //return 0;
}
#endif

void my_free(void *data, void *hint) { free (data); }

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
    int seq = 0;
    while(1) {
        packet_header_t* data = (packet_header_t*)malloc(test_size);
        data->sequence_number = seq;
        data->time = getPreciseTime();
        zmq_msg_t msg;
        zmq_msg_init_data(&msg, data, test_size, my_free, 0);
        int r = zmq_msg_send(&msg, socket, 0);
        if(r < 0) {
            printf("r = %d, %s\n", r, zmq_strerror(zmq_errno()));
        } else {
            printf("Sent 1 packet %d.\n", seq);
        }
        usleep(config.get<int>("zmq.delay", 5000000));
        seq++;
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
    double last_time = 0;
    while(1) {
        zmq_msg_t msg;
        zmq_msg_init(&msg);
        int r = zmq_msg_recv(&msg, socket, 0);
        if(r < 0) {
            printf("r = %d, %s\n", r, zmq_strerror(zmq_errno()));
        } else {
            packet_header_t* data = (packet_header_t*)zmq_msg_data(&msg);
            printf("Message: %10.5lf - %10.5lf - %10.5lf - %d\n", data->time, data->time - getPreciseTime(), data->time - last_time, data->sequence_number);
            last_time = data->time;
        }
        zmq_msg_close(&msg);
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
