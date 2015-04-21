class TimeProfiler {
public:
    class Delegate {
    public:
        virtual void onPrint(const char* scope, const char* name, const char* text, int level);
        virtual ~Delegate() { }
    };
    virtual double getTime() = 0;

    virtual void pushScope(const char* scope) = 0;
    virtual void popScope() = 0;
    virtual void print(const char* text) = 0;
    virtual void setName(const char* name) = 0;

    virtual void setDelegate(Delegate* delegate) = 0;

    virtual ~TimeProfiler() { }
    static TimeProfiler* Create();
    static TimeProfiler* Default();
    static Delegate* STDERR_DELEGATE;
    static Delegate* STDOUT_DELEGATE;
};

class TimeMeasure {
private:
     TimeMeasure() { }
     TimeMeasure(const TimeMeasure& other) { }
     TimeMeasure& operator= (const TimeMeasure&) { return *this; }
     TimeProfiler* profiler;
     double t0;
public:
    TimeMeasure(const char* scope, TimeProfiler* p = 0);
    void begin(const char* task);
    void print(const char* text);
    double done();
    inline double next(const char* task) { double r = done(); begin(task); return r; }
    ~TimeMeasure();
};
