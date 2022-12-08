#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

namespace mfem {

  // Factory to create linear solver by typename or string
  template <typename TypeEnum, typename T, typename... Ts>
  class Factory {
  public:

    // returns a unique_ptr to a new object
    using TypeCreator = std::unique_ptr<T>(*)(Ts... args);

    virtual ~Factory() = default;

    // Find and return optimizer by enumeration type
    std::unique_ptr<T> create(TypeEnum type, Ts... args) {
    
      if (auto it = type_creators_.find(type);
          it !=  type_creators_.end()) {
        return it->second(args...);
      }
      std::cerr << "Factory create: type not found" << std::endl;
      return nullptr;
    }

    const std::vector<std::string>& names() {
      return names_;
    }

    TypeEnum type_by_name(const std::string& name) {
      auto it = str_type_map_.find(name);
      if (it != str_type_map_.end()) {
        return str_type_map_[name];
      }
      std::cerr << "Factory type_by_name: type not found" << std::endl;
      return (TypeEnum)0;
    }

    std::string name_by_type(TypeEnum type) {
      auto it = type_str_map_.find(type);
      if (it != type_str_map_.end()) {
        return type_str_map_[type];
      }
      std::cerr << "Factory name_by_type: name not found" << std::endl;
      return "";
    }

  protected:

    void register_type(TypeEnum type, const std::string& name,
        TypeCreator func) {
      type_creators_.insert(std::pair<TypeEnum, TypeCreator>(type, func));
      str_type_creators_.insert(std::pair<std::string, TypeCreator>(name, func));
      str_type_map_.insert(std::pair<std::string, TypeEnum>(name, type));
      type_str_map_.insert(std::pair<TypeEnum, std::string>(type, name));
      names_.push_back(name);
    }

    std::map<TypeEnum, TypeCreator> type_creators_;
    std::map<std::string, TypeCreator> str_type_creators_;
    std::map<std::string, TypeEnum> str_type_map_;
    std::map<TypeEnum, std::string> type_str_map_;
    std::vector<std::string> names_;
  };
}
