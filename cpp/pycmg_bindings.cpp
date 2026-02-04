#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "osdi_host.h"

namespace py = pybind11;

namespace {

struct ParsedModel {
  std::string name;
  std::unordered_map<std::string, double> params;
};

double parse_number_with_suffix(const std::string &token) {
  std::string s = token;
  double scale = 1.0;
  auto pos = s.find_first_not_of("+-0123456789.eE");
  if (pos != std::string::npos) {
    std::string suffix = s.substr(pos);
    s = s.substr(0, pos);
    std::string suffix_lower;
    suffix_lower.resize(suffix.size());
    std::transform(suffix.begin(), suffix.end(), suffix_lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (suffix_lower == "t") scale = 1e12;
    else if (suffix_lower == "g") scale = 1e9;
    else if (suffix_lower == "meg") scale = 1e6;
    else if (suffix_lower == "k") scale = 1e3;
    else if (suffix_lower == "m") scale = 1e-3;
    else if (suffix_lower == "u") scale = 1e-6;
    else if (suffix_lower == "n") scale = 1e-9;
    else if (suffix_lower == "p") scale = 1e-12;
    else if (suffix_lower == "f") scale = 1e-15;
    else if (suffix_lower == "a") scale = 1e-18;
    else if (suffix_lower == "z") scale = 1e-21;
    else if (suffix_lower == "y") scale = 1e-24;
  }
  if (s.empty() || s == "+" || s == "-") {
    return 0.0;
  }
  return std::stod(s) * scale;
}

std::string to_lower(std::string s) {
  for (auto &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

ParsedModel parse_modelcard(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open modelcard: " + path);
  }
  ParsedModel model;
  std::string line;
  bool in_model = false;
  while (std::getline(file, line)) {
    std::string trimmed = line;
    auto pos_comment = trimmed.find('*');
    if (pos_comment != std::string::npos) {
      trimmed = trimmed.substr(0, pos_comment);
    }
    auto first = trimmed.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      continue;
    }
    trimmed = trimmed.substr(first);
    if (!in_model && trimmed.size() >= 6 && trimmed.substr(0, 6) == ".model") {
      std::istringstream iss(trimmed);
      std::string dot_model;
      std::string model_name;
      std::string model_type;
      iss >> dot_model >> model_name >> model_type;
      model_type = to_lower(model_type);
      if (model_type == "bsimcmg") {
        model.name = model_name;
        in_model = true;
      }
      std::regex assign_re(R"(([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9eE+\-\.]+))");
      for (std::sregex_iterator it(trimmed.begin(), trimmed.end(), assign_re), end; it != end; ++it) {
        std::string key = (*it)[1].str();
        std::string val = (*it)[2].str();
        double parsed = parse_number_with_suffix(val);
        if (to_lower(key) == "eotacc" && parsed < 1.1e-10) {
          parsed = 1.10e-10;
        }
        model.params[key] = parsed;
      }
      continue;
    }
    if (in_model && !trimmed.empty() && trimmed[0] == '+') {
      std::string payload = trimmed.substr(1);
      std::regex assign_re(R"(([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9eE+\-\.]+))");
      for (std::sregex_iterator it(payload.begin(), payload.end(), assign_re), end; it != end; ++it) {
        std::string key = (*it)[1].str();
        std::string val = (*it)[2].str();
        double parsed = parse_number_with_suffix(val);
        if (to_lower(key) == "eotacc" && parsed < 1.1e-10) {
          parsed = 1.10e-10;
        }
        model.params[key] = parsed;
      }
      continue;
    }
  }
  if (model.name.empty()) {
    throw std::runtime_error("no bsimcmg model found in modelcard");
  }
  return model;
}

bool apply_param(
    const OsdiDescriptor *desc,
    osdi_host::OsdiInstance *inst,
    osdi_host::OsdiModel *model,
    const std::string &name,
    double value,
    bool from_modelcard) {
  bool applied = false;
  for (std::uint32_t i = 0; i < desc->num_params; ++i) {
    const OsdiParamOpvar &param = desc->param_opvar[i];
    const char *param_name = param.name ? param.name[0] : nullptr;
    if (!param_name || to_lower(name) != to_lower(param_name)) {
      continue;
    }
    void *ptr = nullptr;
    if ((param.flags & PARA_KIND_MASK) == PARA_KIND_INST) {
      if (!inst) {
        return false;
      }
      ptr = desc->access(inst->data(), model ? model->data() : nullptr, i,
                         ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE);
    } else {
      if (!model) {
        return false;
      }
      ptr = desc->access(nullptr, model->data(), i, ACCESS_FLAG_SET);
    }
    if (!ptr) {
      throw std::runtime_error("invalid parameter access for " + name);
    }
    std::uint32_t ty = (param.flags & PARA_TY_MASK);
    if (ty == PARA_TY_INT) {
      auto *value_ptr = reinterpret_cast<std::int32_t *>(ptr);
      *value_ptr = static_cast<std::int32_t>(value);
      applied = true;
    } else if (ty == PARA_TY_REAL) {
      auto *value_ptr = reinterpret_cast<double *>(ptr);
      *value_ptr = value;
      applied = true;
    } else {
      if (!from_modelcard) {
        throw std::runtime_error("string parameter not supported: " + name);
      }
      applied = true;
    }
    break;
  }
  if (!applied && !from_modelcard) {
    throw std::runtime_error("parameter not found: " + name);
  }
  return applied;
}

class PycmgModel {
public:
  PycmgModel(const std::string &osdi_path,
             const std::string &modelcard_path,
             const std::string &model_name)
      : lib_(osdi_host::OsdiLibrary::load(osdi_path)) {
    if (model_name.empty()) {
      desc_ = lib_.descriptor(0);
    } else {
      desc_ = lib_.descriptor_by_name(model_name);
      if (!desc_) {
        desc_ = lib_.descriptor(0);
      }
    }
    if (!desc_) {
      throw std::runtime_error("OSDI descriptor not found");
    }
    model_ = std::make_unique<osdi_host::OsdiModel>(desc_);

    if (!modelcard_path.empty()) {
      ParsedModel parsed = parse_modelcard(modelcard_path);
      for (const auto &entry : parsed.params) {
        bool ok = apply_param(desc_, nullptr, model_.get(), entry.first, entry.second, true);
        if (!ok) {
          continue;
        }
      }
    }

    model_->process_params();
  }

  const OsdiDescriptor *desc() const { return desc_; }
  osdi_host::OsdiModel &model() { return *model_; }

private:
  osdi_host::OsdiLibrary lib_;
  const OsdiDescriptor *desc_ = nullptr;
  std::unique_ptr<osdi_host::OsdiModel> model_;
};

class PycmgInstance {
public:
  PycmgInstance(std::shared_ptr<PycmgModel> model,
                const py::dict &params,
                double temperature)
      : model_(std::move(model)),
        inst_(model_->desc()),
        temperature_(temperature),
        connected_terminals_(model_->desc()->num_terminals) {
    for (const auto &item : params) {
      const auto key = py::cast<std::string>(item.first);
      const auto value = py::cast<double>(item.second);
      apply_param(model_->desc(), &inst_, &model_->model(), key, value, false);
    }
    inst_.bind_simulation(sim_, model_->model(), connected_terminals_, temperature_);
  }

private:
  std::shared_ptr<PycmgModel> model_;
  osdi_host::OsdiInstance inst_;
  osdi_host::OsdiSimulation sim_;
  double temperature_ = 300.15;
  std::uint32_t connected_terminals_ = 0;
};

}  // namespace

PYBIND11_MODULE(_pycmg, m) {
  m.doc() = "pycmg OSDI bindings";
  py::class_<PycmgModel, std::shared_ptr<PycmgModel>>(m, "Model")
      .def(py::init<const std::string &, const std::string &, const std::string &>(),
           py::arg("osdi_path"),
           py::arg("modelcard_path"),
           py::arg("model_name"));
  py::class_<PycmgInstance>(m, "Instance")
      .def(py::init<std::shared_ptr<PycmgModel>, const py::dict &, double>(),
           py::arg("model"),
           py::arg("params") = py::dict(),
           py::arg("temperature") = 300.15);
}
