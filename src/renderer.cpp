#include "renderer.h"
#include "allosync.h"
#include <stdio.h>
#include <unistd.h>
using namespace std;
using namespace allovolume;

class Delegate : public SyncSystem::Delegate {
public:
    virtual void onMessage(SyncSystem* sync, void* data, size_t length) {
        printf("Message: %s\n", data);
    }
    virtual void onBarrier(SyncSystem* sync, size_t sequence_id) {
        printf("Barrier: %llu\n", sequence_id);
        sync->clearBarrier(sequence_id);
    }
    virtual void onBarrierClear(SyncSystem* sync, size_t sequence_id) {
        printf("Barrier Clear: %llu\n", sequence_id);
    }
};

int main(int argc, char* argv[]) {
    Delegate d;
    if(argc == 1) {
        printf("Server...\n");
        SyncSystem* sync = SyncSystem::Create("allovolume.server.yaml");
        sync->setDelegate(&d);
        sync->start();
        while(1) {
            printf("message...\n");
            sync->sendMessage("Hello World", 12);
            usleep(100000);
            printf("barrier...\n");
            sync->sendBarrier();
            usleep(100000);
        }
    } else {
        SyncSystem* sync = SyncSystem::Create("allovolume.client.yaml");
        sync->setDelegate(&d);
        sync->start();
        while(1) {
            sync->waitEvent();
        }
    }
}
