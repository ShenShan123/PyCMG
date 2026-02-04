#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <complex>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

  void set_params(const py::dict &params, bool allow_rebind) {
    for (const auto &item : params) {
      const auto key = py::cast<std::string>(item.first);
      const auto value = py::cast<double>(item.second);
      apply_param(model_->desc(), &inst_, &model_->model(), key, value, false);
    }

    std::vector<std::uint32_t> internal =
        inst_.process_params(model_->model(), connected_terminals_, temperature_);
    if (internal.size() != sim_.internal_indices.size()) {
      if (!allow_rebind) {
        throw std::runtime_error("topology changed; rebind required");
      }
      sim_ = osdi_host::OsdiSimulation();
      inst_.bind_simulation(sim_, model_->model(), connected_terminals_, temperature_);
    }
  }

  py::dict eval_dc(const py::dict &nodes) {
    set_node_voltages(nodes);
    solve_internal_nodes();

    std::uint32_t flags = ANALYSIS_DC | ANALYSIS_STATIC | CALC_RESIST_JACOBIAN |
                          CALC_RESIST_RESIDUAL | CALC_RESIST_LIM_RHS |
                          CALC_REACT_JACOBIAN | CALC_REACT_RESIDUAL |
                          CALC_REACT_LIM_RHS | CALC_OP | ENABLE_LIM | INIT_LIM;
    (void)inst_.eval(model_->model(), sim_, flags);
    sim_.clear();
    inst_.load_residuals(model_->model(), sim_);
    inst_.load_jacobian(model_->model(), sim_);

    py::dict out;
    out["id"] = read_current("d");
    out["ig"] = read_current("g");
    out["is"] = read_current("s");
    out["ie"] = read_current("e");

    double qg = 0.0, qd = 0.0, qs = 0.0, qb = 0.0;
    read_opvar("qg", "qgate", qg);
    read_opvar("qd", "qdrain", qd);
    read_opvar("qs", "qsource", qs);
    if (!read_opvar("qb", "qbulk", qb)) {
      read_opvar("qe", "qe", qb);
    }
    out["qg"] = qg;
    out["qd"] = qd;
    out["qs"] = qs;
    out["qb"] = qb;

    double gm = 0.0, gds = 0.0, gmb = 0.0;
    read_opvar("gm", "gm", gm);
    read_opvar("gds", "gds", gds);
    if (!read_opvar("gmbs", "gmbs", gmb)) {
      read_opvar("gmb", "gmb", gmb);
    }
    out["gm"] = gm;
    out["gds"] = gds;
    out["gmb"] = gmb;

    auto caps = condense_caps();
    out["cgg"] = caps.cgg;
    out["cgd"] = caps.cgd;
    out["cgs"] = caps.cgs;
    out["cdg"] = caps.cdg;
    out["cdd"] = caps.cdd;

    return out;
  }

private:
  struct CondensedCaps {
    double cgg = 0.0;
    double cgd = 0.0;
    double cgs = 0.0;
    double cdg = 0.0;
    double cdd = 0.0;
  };

  void set_node_voltages(const py::dict &nodes) {
    auto set_or_default = [&](const std::string &name) {
      double value = 0.0;
      py::str key(name);
      if (nodes.contains(key)) {
        value = py::cast<double>(nodes[key]);
      }
      sim_.set_voltage(name, value);
    };
    set_or_default("d");
    set_or_default("g");
    set_or_default("s");
    set_or_default("e");

    if (sim_.node_index.count("di") && !nodes.contains(py::str("di"))) {
      sim_.set_voltage("di", sim_.solve[sim_.node_index["d"]]);
    }
    if (sim_.node_index.count("si") && !nodes.contains(py::str("si"))) {
      sim_.set_voltage("si", sim_.solve[sim_.node_index["s"]]);
    }
  }

  void solve_internal_nodes() {
    bool ok = inst_.solve_internal_nodes(model_->model(), sim_, 200, 1e-9);
    (void)ok;
  }

  double read_current(const std::string &name) const {
    auto it = sim_.node_index.find(name);
    if (it == sim_.node_index.end()) {
      return 0.0;
    }
    return -sim_.residual_resist[it->second];
  }

  bool read_opvar(const std::string &name, const std::string &alias, double &out) const {
    const OsdiDescriptor *desc = model_->desc();
    for (std::uint32_t i = 0; i < desc->num_params; ++i) {
      const OsdiParamOpvar &param = desc->param_opvar[i];
      bool matched = false;
      if (param.name && param.name[0] && to_lower(param.name[0]) == to_lower(name)) {
        matched = true;
      } else if (param.name && param.num_alias > 1) {
        for (std::uint32_t a = 1; a < param.num_alias; ++a) {
          if (param.name[a] && to_lower(param.name[a]) == to_lower(alias)) {
            matched = true;
            break;
          }
        }
      }
      if (!matched) {
        continue;
      }
      void *ptr = desc->access(inst_.data(), model_->model().data(), i,
                               ACCESS_FLAG_READ | ACCESS_FLAG_INSTANCE);
      if (!ptr) {
        ptr = desc->access(inst_.data(), model_->model().data(), i, ACCESS_FLAG_READ);
      }
      if (!ptr) {
        return false;
      }
      std::uint32_t ty = (param.flags & PARA_TY_MASK);
      if (ty == PARA_TY_INT) {
        out = static_cast<double>(*reinterpret_cast<std::int32_t *>(ptr));
      } else if (ty == PARA_TY_REAL) {
        out = *reinterpret_cast<double *>(ptr);
      } else {
        return false;
      }
      return true;
    }
    return false;
  }

  static void build_full_jacobian(
      const osdi_host::OsdiSimulation &sim,
      const std::vector<double> &values,
      std::vector<double> &out) {
    std::size_t n = sim.node_names.size();
    out.assign(n * n, 0.0);
    for (std::size_t k = 0; k < sim.jacobian_info.size(); ++k) {
      auto row = sim.jacobian_info[k].first;
      auto col = sim.jacobian_info[k].second;
      if (row < n && col < n && k < values.size()) {
        out[row * n + col] = values[k];
      }
    }
  }

  static bool solve_linear_system_multi_complex(
      std::vector<std::complex<double>> &a,
      std::vector<std::complex<double>> &b,
      int n,
      int m) {
    for (int i = 0; i < n; ++i) {
      int pivot = i;
      double max_val = std::abs(a[i * n + i]);
      for (int r = i + 1; r < n; ++r) {
        double v = std::abs(a[r * n + i]);
        if (v > max_val) {
          max_val = v;
          pivot = r;
        }
      }
      if (max_val == 0.0) {
        return false;
      }
      if (pivot != i) {
        for (int c = i; c < n; ++c) {
          std::swap(a[i * n + c], a[pivot * n + c]);
        }
        for (int c = 0; c < m; ++c) {
          std::swap(b[i * m + c], b[pivot * m + c]);
        }
      }
      std::complex<double> diag = a[i * n + i];
      for (int c = i; c < n; ++c) {
        a[i * n + c] /= diag;
      }
      for (int c = 0; c < m; ++c) {
        b[i * m + c] /= diag;
      }
      for (int r = i + 1; r < n; ++r) {
        std::complex<double> factor = a[r * n + i];
        if (factor == std::complex<double>(0.0, 0.0)) {
          continue;
        }
        for (int c = i; c < n; ++c) {
          a[r * n + c] -= factor * a[i * n + c];
        }
        for (int c = 0; c < m; ++c) {
          b[r * m + c] -= factor * b[i * m + c];
        }
      }
    }
    for (int i = n - 1; i >= 0; --i) {
      for (int r = 0; r < i; ++r) {
        std::complex<double> factor = a[r * n + i];
        if (factor == std::complex<double>(0.0, 0.0)) {
          continue;
        }
        a[r * n + i] = 0.0;
        for (int c = 0; c < m; ++c) {
          b[r * m + c] -= factor * b[i * m + c];
        }
      }
    }
    return true;
  }

  static bool condense_capacitance(
      const std::vector<double> &g_full,
      const std::vector<double> &c_full,
      std::size_t full_size,
      const std::vector<std::uint32_t> &external,
      const std::vector<std::uint32_t> &internal,
      std::vector<double> &c_condensed) {
    std::size_t ne = external.size();
    std::size_t ni = internal.size();
    c_condensed.assign(ne * ne, 0.0);
    if (ne == 0) {
      return true;
    }
    const std::complex<double> jw(0.0, 1.0);
    auto idx = [&](std::uint32_t r, std::uint32_t c) {
      return static_cast<std::size_t>(r) * full_size + static_cast<std::size_t>(c);
    };

    std::vector<std::complex<double>> yee(ne * ne);
    std::vector<std::complex<double>> yei(ne * ni);
    std::vector<std::complex<double>> yie(ni * ne);
    std::vector<std::complex<double>> yii(ni * ni);

    for (std::size_t r = 0; r < ne; ++r) {
      for (std::size_t c = 0; c < ne; ++c) {
        std::size_t pos = idx(external[r], external[c]);
        yee[r * ne + c] = g_full[pos] + jw * c_full[pos];
      }
      for (std::size_t c = 0; c < ni; ++c) {
        std::size_t pos = idx(external[r], internal[c]);
        yei[r * ni + c] = g_full[pos] + jw * c_full[pos];
      }
    }
    for (std::size_t r = 0; r < ni; ++r) {
      for (std::size_t c = 0; c < ne; ++c) {
        std::size_t pos = idx(internal[r], external[c]);
        yie[r * ne + c] = g_full[pos] + jw * c_full[pos];
      }
      for (std::size_t c = 0; c < ni; ++c) {
        std::size_t pos = idx(internal[r], internal[c]);
        yii[r * ni + c] = g_full[pos] + jw * c_full[pos];
      }
    }

    if (ni == 0) {
      for (std::size_t r = 0; r < ne; ++r) {
        for (std::size_t c = 0; c < ne; ++c) {
          c_condensed[r * ne + c] = std::imag(yee[r * ne + c]);
        }
      }
      return true;
    }

    std::vector<std::complex<double>> yii_copy = yii;
    std::vector<std::complex<double>> yie_copy = yie;
    if (!solve_linear_system_multi_complex(yii_copy, yie_copy,
                                           static_cast<int>(ni),
                                           static_cast<int>(ne))) {
      return false;
    }

    for (std::size_t r = 0; r < ne; ++r) {
      for (std::size_t c = 0; c < ne; ++c) {
        std::complex<double> accum = yee[r * ne + c];
        for (std::size_t k = 0; k < ni; ++k) {
          accum -= yei[r * ni + k] * yie_copy[k * ne + c];
        }
        c_condensed[r * ne + c] = std::imag(accum);
      }
    }
    return true;
  }

  CondensedCaps condense_caps() const {
    std::vector<double> g_full;
    std::vector<double> c_full;
    build_full_jacobian(sim_, sim_.jacobian_resist, g_full);
    build_full_jacobian(sim_, sim_.jacobian_react, c_full);
    std::vector<double> c_condensed;
    condense_capacitance(g_full, c_full, sim_.node_names.size(),
                         sim_.terminal_indices, sim_.internal_indices, c_condensed);
    auto idx_of = [&](const std::string &name) -> int {
      for (std::size_t i = 0; i < sim_.terminal_indices.size(); ++i) {
        if (sim_.node_names[sim_.terminal_indices[i]] == name) {
          return static_cast<int>(i);
        }
      }
      return -1;
    };
    CondensedCaps caps;
    if (c_condensed.empty()) {
      return caps;
    }
    int g = idx_of("g");
    int d = idx_of("d");
    int s = idx_of("s");
    if (g >= 0) {
      std::size_t n = sim_.terminal_indices.size();
      caps.cgg = c_condensed[g * n + g];
      if (d >= 0) caps.cgd = c_condensed[g * n + d];
      if (s >= 0) caps.cgs = c_condensed[g * n + s];
    }
    if (d >= 0 && g >= 0) {
      std::size_t n = sim_.terminal_indices.size();
      caps.cdg = c_condensed[d * n + g];
      caps.cdd = c_condensed[d * n + d];
    }
    return caps;
  }

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
           py::arg("temperature") = 300.15)
      .def("set_params", &PycmgInstance::set_params,
           py::arg("params"),
           py::arg("allow_rebind") = false)
      .def("eval_dc", &PycmgInstance::eval_dc, py::arg("nodes"));
}
