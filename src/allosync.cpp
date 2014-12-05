#include <zmq.h>
#include <unistd.h>
#include <sys/time.h>
#include <map>
#include <set>
#include <string>
#include <pthread.h>
#include <string.h>
#include "configparser.h"
#include "allosync.h"

using namespace std;

namespace allovolume {

    class SyncSystemImpl : public SyncSystem {
    public:

        static const unsigned char TYPE_MESSAGE            = 1;
        static const unsigned char TYPE_BARRIER_REQUEST    = 2;
        static const unsigned char TYPE_BARRIER_RESPONSE   = 3;
        static const unsigned char TYPE_REGISTER_CLIENT    = 4;
        static const unsigned char TYPE_TIME               = 5;

        struct MessageHeader {
            size_t sequence_number;
            unsigned char type;
        };

        struct BarrierInfo {
            size_t client_count;
            size_t replied_clients;
        };

        struct BarrierMessage {
            char client_name[64];
            size_t sequence_number;
        };

        SyncSystemImpl(const char* configfile) {
            delegate = &empty_delegate;
            char hostname[256];
            gethostname(hostname, 256);
            config.parseFile(configfile, hostname);
            config.parseFile(configfile);
            is_server = (config.get<string>("SyncSystem.role") == "server");
            sequence_number = 0;
            sprintf(client_name, "%s-%10.5lf", hostname, getPreciseTime());
        }

        void _startup() {
            zmq_context = zmq_ctx_new();

            if(is_server) {
                // Server.
                // Pubsub socket.
                socket_pubsub = zmq_socket(zmq_context, ZMQ_PUB);
                int value;
                value = config.get<int>("SyncSystem.zmq.sndhwm", 10000);
                zmq_setsockopt(socket_pubsub, ZMQ_SNDHWM, &value, sizeof(int));
                value = config.get<int>("SyncSystem.zmq.sndbuf", 0);
                zmq_setsockopt(socket_pubsub, ZMQ_SNDBUF, &value, sizeof(int));
                value = config.get<int>("SyncSystem.zmq.rate", 10000000);
                zmq_setsockopt(socket_pubsub, ZMQ_RATE, &value, sizeof(int));

                zmq_bind(socket_pubsub, config.get<string>("SyncSystem.broadcast").c_str());

                pthread_t th;
                pthread_create(&th, NULL, server_thread_entry, this);

            } else {
                // Client.
                // Pubsub socket.
                socket_pubsub = zmq_socket(zmq_context, ZMQ_SUB);
                int value;
                value = config.get<int>("SyncSystem.zmq.rcvhwm", 10000);
                zmq_setsockopt(socket_pubsub, ZMQ_RCVHWM, &value, sizeof(int));
                value = config.get<int>("SyncSystem.zmq.rcvbuf", 0);
                zmq_setsockopt(socket_pubsub, ZMQ_RCVBUF, &value, sizeof(int));
                value = config.get<int>("SyncSystem.zmq.rate", 10000000);
                zmq_setsockopt(socket_pubsub, ZMQ_RATE, &value, sizeof(int));

                zmq_connect(socket_pubsub, config.get<string>("SyncSystem.broadcast").c_str());
                zmq_setsockopt(socket_pubsub, ZMQ_SUBSCRIBE, "", 0);

                socket_feedback = zmq_socket(zmq_context, ZMQ_PUSH);
                zmq_connect(socket_feedback, config.get<string>("SyncSystem.feedback").c_str());

                _register_client();
            }
        }

        static void* server_thread_entry(void* self) {
            ((SyncSystemImpl*)self)->server_thread();
        }

        void server_thread() {
            socket_feedback = zmq_socket(zmq_context, ZMQ_PULL);
            int r = zmq_bind(socket_feedback, config.get<string>("SyncSystem.feedback").c_str());
            if(r < 0) {
                fprintf(stderr, "SyncSystem::zmq_bind: %s\n", zmq_strerror(zmq_errno()));
                return;
            }
            zmq_setsockopt(socket_feedback, ZMQ_SUBSCRIBE, "", 0);

            while(1) {
                zmq_msg_t msg;
                zmq_msg_init(&msg);
                int r = zmq_msg_recv(&msg, socket_feedback, 0);
                if(r < 0) {
                    fprintf(stderr, "SyncSystem::waitEvent: %s\n", zmq_strerror(zmq_errno()));
                } else {
                    MessageHeader& hdr = *(MessageHeader*)zmq_msg_data(&msg);
                    if(hdr.type == TYPE_BARRIER_RESPONSE) {
                        BarrierMessage& bmsg = *(BarrierMessage*)(((MessageHeader*)zmq_msg_data(&msg)) + 1);
                        if(barriers.find(bmsg.sequence_number) != barriers.end()) {
                            barriers[bmsg.sequence_number].replied_clients += 1;
                            if(barriers[bmsg.sequence_number].replied_clients >= barriers[bmsg.sequence_number].client_count) {
                                delegate->onBarrierClear(this, bmsg.sequence_number);
                                barriers.erase(bmsg.sequence_number);
                            }
                        }

                    } else if(hdr.type == TYPE_REGISTER_CLIENT) {
                        BarrierMessage& bmsg = *(BarrierMessage*)(((MessageHeader*)zmq_msg_data(&msg)) + 1);
                        clients.insert(bmsg.client_name);
                    }
                }
                zmq_msg_close(&msg);
            }
        }

        static void my_free(void *data, void *hint) { free(data); }
        static void null_free(void *data, void *hint) { }

        virtual void sendMessage(const void* data, size_t length) {
            if(is_server) {
                zmq_msg_t msg;
                MessageHeader* hdr = (MessageHeader*)malloc(length + sizeof(MessageHeader));;
                hdr->type = TYPE_MESSAGE;
                hdr->sequence_number = sequence_number;
                sequence_number += 1;
                memcpy(hdr + 1, data, length);
                zmq_msg_init_data(&msg, hdr, length + sizeof(MessageHeader), my_free, 0);
                int r = zmq_msg_send(&msg, socket_pubsub, 0);
                if(r < 0) {
                    fprintf(stderr, "SyncSystem::sendMessage: %s\n", zmq_strerror(zmq_errno()));
                }
            }
        }

        virtual size_t sendBarrier() {
            if(is_server) {
                if(clients.size() == 0) {
                    delegate->onBarrierClear(this, sequence_number);
                    return sequence_number++;
                }
                zmq_msg_t msg;
                MessageHeader* hdr = (MessageHeader*)malloc(sizeof(MessageHeader));
                hdr->type = TYPE_BARRIER_REQUEST;
                size_t result = hdr->sequence_number = sequence_number;
                barriers[sequence_number].replied_clients = 0;
                barriers[sequence_number].client_count = clients.size();
                sequence_number += 1;
                zmq_msg_init_data(&msg, hdr, sizeof(MessageHeader), my_free, 0);
                int r = zmq_msg_send(&msg, socket_pubsub, 0);
                if(r < 0) {
                    fprintf(stderr, "SyncSystem::sendBarrier: %s\n", zmq_strerror(zmq_errno()));
                }
                return result;
            }
            return 0;
        }

        virtual void sendTime() {
            if(is_server) {
                zmq_msg_t msg;
                MessageHeader* hdr = (MessageHeader*)malloc(sizeof(MessageHeader) + sizeof(double));
                hdr->type = TYPE_TIME;
                hdr->sequence_number = sequence_number;
                *(double*)(hdr + 1) = getPreciseTime();
                sequence_number += 1;
                zmq_msg_init_data(&msg, hdr, sizeof(MessageHeader) + sizeof(double), my_free, 0);
                int r = zmq_msg_send(&msg, socket_pubsub, 0);
                if(r < 0) {
                    fprintf(stderr, "SyncSystem::sendTime: %s\n", zmq_strerror(zmq_errno()));
                }
            }
        }

        virtual void clearBarrier(size_t sequence_id) {
            zmq_msg_t msg;
            MessageHeader* hdr = (MessageHeader*)malloc(sizeof(MessageHeader) + sizeof(BarrierMessage));
            hdr->type = TYPE_BARRIER_RESPONSE;
            hdr->sequence_number = sequence_id;
            BarrierMessage* binfo = (BarrierMessage*)(hdr + 1);
            binfo->sequence_number = sequence_id;
            strcpy(binfo->client_name, client_name);
            zmq_msg_init_data(&msg, hdr, sizeof(MessageHeader) + sizeof(BarrierMessage), my_free, 0);
            int r = zmq_msg_send(&msg, socket_feedback, 0);
            if(r < 0) {
                fprintf(stderr, "SyncSystem::clearBarrier: %s\n", zmq_strerror(zmq_errno()));
            }
        }

        void _register_client() {
            void* data = malloc(sizeof(MessageHeader) + sizeof(BarrierMessage));
            MessageHeader& hdr = *(MessageHeader*)data;
            hdr.type = TYPE_REGISTER_CLIENT;
            hdr.sequence_number = 0;
            BarrierMessage& binfo = *(BarrierMessage*)(((MessageHeader*)data) + 1);
            binfo.sequence_number = 0;
            strcpy(binfo.client_name, client_name);
            zmq_msg_t msg;
            zmq_msg_init_data(&msg, data, sizeof(MessageHeader) + sizeof(BarrierMessage), my_free, 0);
            int r = zmq_msg_send(&msg, socket_feedback, 0);
            if(r < 0) {
                fprintf(stderr, "SyncSystem::_register_client: %s\n", zmq_strerror(zmq_errno()));
            }
        }

        virtual void waitEvent() {
            if(!is_server) {
                zmq_msg_t msg;
                zmq_msg_init(&msg);
                int r = zmq_msg_recv(&msg, socket_pubsub, 0);
                if(r < 0) {
                    fprintf(stderr, "SyncSystem::waitEvent: %s\n", zmq_strerror(zmq_errno()));
                } else {
                    MessageHeader* hdr = (MessageHeader*)zmq_msg_data(&msg);
                    void* data = hdr + 1;
                    size_t data_length = zmq_msg_size(&msg) - sizeof(MessageHeader);
                    if(hdr->type == TYPE_MESSAGE) {
                        delegate->onMessage(this, data, data_length);
                    } else if(hdr->type == TYPE_TIME) {
                        time_diff = *(double*)data - getPreciseTime();
                    } else {
                        delegate->onBarrier(this, hdr->sequence_number);
                    }
                }
                zmq_msg_close(&msg);
            }
        }

        virtual void startInThread() {

        }

        virtual void start() {
            _startup();
        }

        virtual void setDelegate(Delegate* delegate_) {
            delegate = delegate_;
            if(!delegate) delegate = &empty_delegate;
        }

        double getPreciseTime() {
            timeval t;
            gettimeofday(&t, 0);
            double s = t.tv_sec;
            s += t.tv_usec / 1000000.0;
            return s;
        }

        virtual double getLocalTime() {
            return getPreciseTime();
        }

        virtual double getTime() {
            if(is_server) return getPreciseTime();
            else return getPreciseTime() + time_diff;
        }

        virtual bool isServer() {
            return is_server;
        }

        ConfigParser config;

        void* zmq_context;
        void* socket_pubsub;
        void* socket_feedback;

        bool is_server;

        Delegate* delegate;
        Delegate empty_delegate;

        size_t sequence_number;
        double time_diff;
        map<size_t, BarrierInfo> barriers;
        set<string> clients;

        char client_name[64];
    };

    void SyncSystem::Delegate::onMessage(SyncSystem* sync, void* data, size_t length) { }
    void SyncSystem::Delegate::onBarrier(SyncSystem* sync, size_t sequence_id) {
        sync->clearBarrier(sequence_id);
    }
    void SyncSystem::Delegate::onBarrierClear(SyncSystem* sync, size_t sequence_id) { }

    SyncSystem* SyncSystem::Create(const char* configfile) {
        return new SyncSystemImpl(configfile);
    }
}
