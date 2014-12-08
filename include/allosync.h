#ifndef ALLOSYNC_H_INCLUDED
#define ALLOSYNC_H_INCLUDED

// Synchronization kernel for Allosphere integration

namespace allovolume {
    // Server:                      Client:
    //   broadcastMessage(...)        !onMessage(...)
    //   broadcastMessage(...)        !onMessage(...)
    //   broadcastBarrier()           !onBarrier(seq_id)
    //                                clearBarrier(seq_id)
    //   !onBarrierClear(seq_id)
    //   ...

    class SyncSystem {
    public:

        class Delegate {
        public:
            // Client events:
            virtual void onMessage(SyncSystem* sync, void* data, size_t length);
            virtual void onBarrier(SyncSystem* sync, size_t sequence_id);
            // Server events:
            virtual void onBarrierClear(SyncSystem* sync, size_t sequence_id);
            virtual ~Delegate() { }
        };

        // Determine the role of the current instance.
        virtual bool isServer() = 0;

        // Server:
        virtual void sendMessage(const void* data, size_t length) = 0;
        virtual void sendTime() = 0;
        virtual size_t sendBarrier() = 0;

        // Client:
        virtual void clearBarrier(size_t sequence_id) = 0;

        // Start and wait for events manually.
        virtual void start() = 0;
        virtual void waitEvent() = 0;

        // Get synchronized time.
        virtual double getTime() = 0;
        virtual double getLocalTime() = 0;

        virtual void setDelegate(Delegate* delegate) = 0;

        virtual void* getZMQContext() = 0;

        virtual ~SyncSystem() { }

        static SyncSystem* Create(const char* configfile);
    };
}

#endif
