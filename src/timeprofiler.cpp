#include "timeprofiler.h"

#include <stdio.h>
#include <string>
#include <deque>
#include <sys/time.h>

using namespace std;

namespace {
    class StderrDelegate : public TimeProfiler::Delegate {
    public:
        StderrDelegate(FILE* fp_) { fp = fp_; }

        virtual void onPrint(const char* scope, const char* name, const char* text, int level) {
            for(int i = 0; i < level - 1; i++)
                fprintf(fp, "  ");
            fprintf(fp, "[%s] %s: %s\n", scope, name, text);
        }

        FILE* fp;
    };

    StderrDelegate stderr_delegate(stderr);
    StderrDelegate stdout_delegate(stdout);
}

TimeProfiler::Delegate* TimeProfiler::STDERR_DELEGATE = &stderr_delegate;
TimeProfiler::Delegate* TimeProfiler::STDOUT_DELEGATE = &stdout_delegate;

class TimeProfilerImpl : public TimeProfiler {
public:
    struct ScopeData {
        string scope;
        string name;
    };

    TimeProfilerImpl() {
        delegate = NULL;
    }

    virtual void setDelegate(Delegate* delegate_) {
        delegate = delegate_;
    }

    virtual double getTime() {
        timeval t;
        gettimeofday(&t, 0);
        double s = t.tv_sec;
        s += t.tv_usec / 1000000.0;
        return s;
    }

    virtual void pushScope(const char* scope) {
        ScopeData s;
        s.scope = scope;
        s.name = "scope";
        scope_stack.push_back(s);
    }
    virtual void popScope() {
        scope_stack.pop_back();
    }
    virtual void print(const char* text) {
        if(delegate) {
            delegate->onPrint(scope_stack.back().scope.c_str(), scope_stack.back().name.c_str(), text, scope_stack.size());
        }
    }
    virtual void setName(const char* name_) {
        scope_stack.back().name = name_;
    }

    static TimeProfilerImpl* Global() {
        if(!global_instance) global_instance = new TimeProfilerImpl();
        return global_instance;
    }

    Delegate* delegate;
    static TimeProfilerImpl* global_instance;

    deque<ScopeData> scope_stack;
};

TimeProfilerImpl* TimeProfilerImpl::global_instance = NULL;

void TimeProfiler::Delegate::onPrint(const char* scope, const char* name, const char* text, int level) { }

TimeProfiler* TimeProfiler::Create() {
    return new TimeProfilerImpl();
}

TimeProfiler* TimeProfiler::Default() {
    return TimeProfilerImpl::Global();
}

TimeMeasure::TimeMeasure(const char* scope, TimeProfiler* p) {
    if(p) profiler = p;
    else profiler = TimeProfilerImpl::Default();
    profiler->pushScope(scope);
    profiler->setName("scope");
    profiler->print("Enter");
}
void TimeMeasure::begin(const char* name) {
    profiler->setName(name);
    t0 = profiler->getTime();
}
void TimeMeasure::print(const char* text) {
    profiler->print(text);
}
double TimeMeasure::done() {
    double t = profiler->getTime() - t0;
    char buf[64];
    sprintf(buf, "%.3lf ms", t * 1000);
    profiler->print(buf);
    return t;
}
TimeMeasure::~TimeMeasure() {
    profiler->popScope();
}
