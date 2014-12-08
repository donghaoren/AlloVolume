#ifndef ALLOVOLUME_CONFIG_PARSER_H_INCLUDED
#define ALLOVOLUME_CONFIG_PARSER_H_INCLUDED

#include <vector>
#include <yaml-cpp/yaml.h>
#include <iostream>

class ConfigParser {
public:
    ConfigParser() { }
    ConfigParser(const std::string& path) {
        parseFile(path);
    }

    void parseFile(const std::string& path) {
        entries.push_back(YAML::LoadFile(path));
    }

    void parseFile(const std::string& path, const std::string& key) {
        YAML::Node part = YAML::LoadFile(path)[key];
        if(!part.IsNull() && part.IsDefined())
            entries.push_back(part);
    }

    template<typename T>
    T get(const std::string& path, const T& fallback = T()) {
        for(size_t i = 0; i < entries.size(); i++) {
            YAML::Node node = entries[i];
            size_t pos = 0;
            while(pos < path.size()) {
                std::string subpath;
                size_t end = path.find('.', pos);
                if(end == std::string::npos) {
                    subpath = path.substr(pos);
                    pos = path.size();
                } else {
                    subpath = path.substr(pos, end - pos);
                    pos = end + 1;
                }
                if(!node.IsNull()) {
                    node.reset(node[subpath]);
                }
            }
            if(!node.IsNull() && node.IsDefined()) {
                return node.as<T>();
            }
        }
        return fallback;
    }

    std::vector<YAML::Node> entries;
};

#endif
