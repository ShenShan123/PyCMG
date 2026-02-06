#include "osdi_host.h"

#include <cctype>
#include <cstdint>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct NodeValue {
  std::string name;
  double value = 0.0;
};

struct ParamValue {
  std::string name;
  double value = 0.0;
  bool from_modelcard = false;
};

struct ParsedModel {
  std::string name;
  std::unordered_map<std::string, double> params;
};

void print_usage(const char *argv0) {
  std::cout
      << "Usage: " << argv0
      << " [--osdi PATH] [--model NAME] [--temp K] [--node name=val ...]"
      << " [--param name=val ...] [--modelcard PATH]"
      << " [--list-nodes] [--list-params] [--no-solve-internal]"
      << " [--dump-param NAME ...] [--print-charges] [--print-cap] [--print-derivs] [--quiet]\n";
}

bool parse_node_value(const std::string &arg, NodeValue &out) {
  auto pos = arg.find('=');
  if (pos == std::string::npos || pos == 0 || pos + 1 >= arg.size()) {
    return false;
  }
  out.name = arg.substr(0, pos);
  try {
    out.value = std::stod(arg.substr(pos + 1));
  } catch (...) {
    return false;
  }
  return true;
}

bool parse_param_value(const std::string &arg, ParamValue &out) {
  auto pos = arg.find('=');
  if (pos == std::string::npos || pos == 0 || pos + 1 >= arg.size()) {
    return false;
  }
  out.name = arg.substr(0, pos);
  try {
    out.value = std::stod(arg.substr(pos + 1));
  } catch (...) {
    return false;
  }
  return true;
}

void list_nodes(const OsdiDescriptor *desc) {
  if (!desc || !desc->nodes) {
    return;
  }
  std::cout << "Nodes (" << desc->num_nodes << ") terminals=" << desc->num_terminals << "\n";
  for (std::uint32_t i = 0; i < desc->num_nodes; ++i) {
    const auto &node = desc->nodes[i];
    std::cout << "  [" << i << "] " << (node.name ? node.name : "(null)")
              << (i < desc->num_terminals ? " (terminal)" : "") << "\n";
  }
}

void list_params(const OsdiDescriptor *desc) {
  if (!desc || !desc->param_opvar) {
    return;
  }
  std::cout << "Params (" << desc->num_params << ")\n";
  for (std::uint32_t i = 0; i < desc->num_params; ++i) {
    const auto &param = desc->param_opvar[i];
    const char *name = param.name ? param.name[0] : "(null)";
    const char *kind = "model";
    if ((param.flags & PARA_KIND_MASK) == PARA_KIND_INST) {
      kind = "instance";
    } else if ((param.flags & PARA_KIND_MASK) == PARA_KIND_OPVAR) {
      kind = "opvar";
    }
    std::cout << "  [" << i << "] " << name << " (" << kind << ")\n";
  }
}

double parse_number_with_suffix(const std::string &token) {
  std::string s = token;
  if (s.empty()) {
    return 0.0;
  }
  for (auto &c : s) {
    c = static_cast<char>(std::tolower(c));
  }
  double scale = 1.0;
  if (s.size() >= 3 && s.compare(s.size() - 3, 3, "meg") == 0) {
    scale = 1e6;
    s.resize(s.size() - 3);
  } else if (s.back() == 't') {
    scale = 1e12;
    s.pop_back();
  } else if (s.back() == 'g') {
    scale = 1e9;
    s.pop_back();
  } else if (s.back() == 'k') {
    scale = 1e3;
    s.pop_back();
  } else if (s.back() == 'm') {
    scale = 1e-3;
    s.pop_back();
  } else if (s.back() == 'u') {
    scale = 1e-6;
    s.pop_back();
  } else if (s.back() == 'n') {
    scale = 1e-9;
    s.pop_back();
  } else if (s.back() == 'p') {
    scale = 1e-12;
    s.pop_back();
  } else if (s.back() == 'f') {
    scale = 1e-15;
    s.pop_back();
  } else if (s.back() == 'a') {
    scale = 1e-18;
    s.pop_back();
  }
  if (s.empty()) {
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

bool read_opvar_value(
    const OsdiDescriptor *descriptor,
    void *inst_data,
    void *model_data,
    const std::string &name,
    double *out_value) {
  if (!descriptor || !descriptor->param_opvar || !out_value) {
    return false;
  }
  auto name_lower = to_lower(name);
  std::uint32_t total = descriptor->num_params + descriptor->num_opvars;
  for (std::uint32_t i = 0; i < total; ++i) {
    const OsdiParamOpvar &param = descriptor->param_opvar[i];
    if ((param.flags & PARA_KIND_MASK) != PARA_KIND_OPVAR) {
      continue;
    }
    bool matched = false;
    if (param.num_alias == 0) {
      const char *alias_name = param.name ? param.name[0] : nullptr;
      if (alias_name && to_lower(alias_name) == name_lower) {
        matched = true;
      }
    } else {
      for (std::uint32_t alias = 0; alias < param.num_alias; ++alias) {
        const char *alias_name = param.name ? param.name[alias] : nullptr;
        if (!alias_name) {
          continue;
        }
        if (to_lower(alias_name) == name_lower) {
          matched = true;
          break;
        }
      }
    }
    if (!matched) {
      continue;
    }
    void *ptr = descriptor->access(inst_data, model_data, i, ACCESS_FLAG_INSTANCE);
    if (!ptr) {
      ptr = descriptor->access(inst_data, model_data, i, ACCESS_FLAG_READ | ACCESS_FLAG_INSTANCE);
    }
    if (!ptr) {
      return false;
    }
    std::uint32_t ty = (param.flags & PARA_TY_MASK);
    if (ty == PARA_TY_INT) {
      *out_value = static_cast<double>(*reinterpret_cast<std::int32_t *>(ptr));
      return true;
    }
    if (ty == PARA_TY_REAL) {
      *out_value = *reinterpret_cast<double *>(ptr);
      return true;
    }
    return false;
  }
  return false;
}

void build_full_jacobian(
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

bool solve_linear_system_multi_complex(
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

bool condense_capacitance(
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
  std::vector<std::complex<double>> x = yie;
  if (!solve_linear_system_multi_complex(yii_copy, x, static_cast<int>(ni), static_cast<int>(ne))) {
    return false;
  }

  for (std::size_t r = 0; r < ne; ++r) {
    for (std::size_t c = 0; c < ne; ++c) {
      std::complex<double> sum(0.0, 0.0);
      for (std::size_t k = 0; k < ni; ++k) {
        sum += yei[r * ni + k] * x[k * ne + c];
      }
      std::complex<double> y = yee[r * ne + c] - sum;
      c_condensed[r * ne + c] = std::imag(y);
    }
  }
  return true;
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

}  // namespace

int main(int argc, char **argv) {
  std::string osdi_path = "build/osdi/bsimcmg.osdi";
  std::string model_name;
  double temp_k = 300.15;
  bool list_only = false;
  bool list_params_only = false;
  std::vector<NodeValue> node_values;
  std::vector<ParamValue> param_values;
  std::string modelcard_path;
  bool solve_internal = true;
  bool print_charges = false;
  bool print_cap = false;
  bool print_derivs = false;
  bool quiet = false;
  std::unordered_map<std::string, bool> provided_nodes;
  std::vector<std::string> dump_params;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--osdi" && i + 1 < argc) {
      osdi_path = argv[++i];
    } else if (arg == "--model" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (arg == "--temp" && i + 1 < argc) {
      temp_k = std::stod(argv[++i]);
    } else if (arg == "--node" && i + 1 < argc) {
      NodeValue nv;
      if (!parse_node_value(argv[++i], nv)) {
        std::cerr << "Invalid --node value\n";
        return 1;
      }
      node_values.push_back(nv);
    } else if (arg == "--param" && i + 1 < argc) {
      ParamValue pv;
      if (!parse_param_value(argv[++i], pv)) {
        std::cerr << "Invalid --param value\n";
        return 1;
      }
      pv.from_modelcard = false;
      param_values.push_back(pv);
    } else if (arg == "--modelcard" && i + 1 < argc) {
      modelcard_path = argv[++i];
    } else if (arg == "--list-nodes") {
      list_only = true;
    } else if (arg == "--list-params") {
      list_params_only = true;
    } else if (arg == "--no-solve-internal") {
      solve_internal = false;
    } else if (arg == "--dump-param" && i + 1 < argc) {
      dump_params.push_back(argv[++i]);
    } else if (arg == "--print-charges") {
      print_charges = true;
    } else if (arg == "--print-cap") {
      print_cap = true;
    } else if (arg == "--print-derivs") {
      print_derivs = true;
    } else if (arg == "--quiet") {
      quiet = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  try {
    auto lib = osdi_host::OsdiLibrary::load(osdi_path);
    const OsdiDescriptor *desc =
        model_name.empty() ? lib.descriptor(0) : lib.descriptor_by_name(model_name);
    if (!desc) {
      throw std::runtime_error("OSDI descriptor not found");
    }

    if (list_only) {
      list_nodes(desc);
      return 0;
    }
    if (list_params_only) {
      list_params(desc);
      return 0;
    }

    osdi_host::OsdiModel model(desc);
    osdi_host::OsdiInstance inst(desc);

    if (!modelcard_path.empty()) {
      ParsedModel parsed = parse_modelcard(modelcard_path);
      if (!model_name.empty() && parsed.name != model_name) {
        throw std::runtime_error("modelcard model name mismatch");
      }
      for (const auto &entry : parsed.params) {
        param_values.push_back({entry.first, entry.second, true});
      }
    }

    for (const auto &pv : param_values) {
      bool applied = false;
      for (std::uint32_t i = 0; i < desc->num_params; ++i) {
        const OsdiParamOpvar &param = desc->param_opvar[i];
        const char *param_name = param.name ? param.name[0] : nullptr;
        if (!param_name || to_lower(pv.name) != to_lower(param_name)) {
          continue;
        }
        void *ptr = nullptr;
        if ((param.flags & PARA_KIND_MASK) == PARA_KIND_INST) {
          ptr = desc->access(inst.data(), model.data(), i, ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE);
        } else {
          ptr = desc->access(nullptr, model.data(), i, ACCESS_FLAG_SET);
        }
        if (!ptr) {
          throw std::runtime_error("invalid parameter access for " + pv.name);
        }
        std::uint32_t ty = (param.flags & PARA_TY_MASK);
        if (ty == PARA_TY_INT) {
          auto *value_ptr = reinterpret_cast<std::int32_t *>(ptr);
          *value_ptr = static_cast<std::int32_t>(pv.value);
        } else if (ty == PARA_TY_REAL) {
          auto *value_ptr = reinterpret_cast<double *>(ptr);
          *value_ptr = pv.value;
        } else {
          std::cerr << "Warning: string param not supported: " << pv.name << "\n";
          applied = true;
          break;
        }
        applied = true;
        break;
      }
      if (!applied) {
        if (pv.from_modelcard) {
          std::cerr << "Warning: parameter not found in OSDI model: " << pv.name << "\n";
          continue;
        }
        throw std::runtime_error("parameter not found: " + pv.name);
      }
    }

    model.process_params();

    if (!dump_params.empty()) {
      for (const auto &name : dump_params) {
        bool found = false;
        for (std::uint32_t i = 0; i < desc->num_params; ++i) {
          const OsdiParamOpvar &param = desc->param_opvar[i];
          const char *param_name = param.name ? param.name[0] : nullptr;
          if (!param_name || to_lower(name) != to_lower(param_name)) {
            continue;
          }
          void *ptr = nullptr;
          if ((param.flags & PARA_KIND_MASK) == PARA_KIND_INST) {
            ptr = desc->access(inst.data(), model.data(), i, ACCESS_FLAG_INSTANCE);
          } else {
            ptr = desc->access(nullptr, model.data(), i, ACCESS_FLAG_READ);
          }
          if (!ptr) {
            std::cerr << "Param access failed: " << name << "\n";
            found = true;
            break;
          }
          std::uint32_t ty = (param.flags & PARA_TY_MASK);
          const char *kind = "model";
          if ((param.flags & PARA_KIND_MASK) == PARA_KIND_INST) {
            kind = "instance";
          } else if ((param.flags & PARA_KIND_MASK) == PARA_KIND_OPVAR) {
            kind = "opvar";
          }
          if (ty == PARA_TY_INT) {
            auto value = *reinterpret_cast<std::int32_t *>(ptr);
            std::cout << "Param " << param_name << " = " << value
                      << " (" << kind << ", int)\n";
          } else if (ty == PARA_TY_REAL) {
            double value = *reinterpret_cast<double *>(ptr);
            std::cout << "Param " << param_name << " = " << std::setprecision(12) << value
                      << " (" << kind << ", real)\n";
          } else {
            std::cout << "Param " << param_name << " = (string) (" << kind << ", str)\n";
          }
          found = true;
          break;
        }
        if (!found) {
          std::cerr << "Param not found: " << name << "\n";
        }
      }
    }

    osdi_host::OsdiSimulation sim;
    inst.bind_simulation(sim, model, desc->num_terminals, temp_k);

    for (const auto &nv : node_values) {
      sim.set_voltage(nv.name, nv.value);
      provided_nodes[to_lower(nv.name)] = true;
    }

    if (solve_internal) {
      auto set_if_missing = [&](const std::string &name, const std::string &fallback) {
        if (provided_nodes.find(to_lower(name)) != provided_nodes.end()) {
          return;
        }
        auto it_fb = sim.node_index.find(fallback);
        if (it_fb == sim.node_index.end()) {
          return;
        }
        sim.set_voltage(name, sim.solve[it_fb->second]);
      };
      set_if_missing("di", "d");
      set_if_missing("si", "s");
    }

    if (solve_internal) {
      bool ok = inst.solve_internal_nodes(model, sim, 200, 1e-9);
      if (!ok) {
        std::cerr << "Warning: internal node solver did not converge\n";
      }
    }

    std::uint32_t flags = ANALYSIS_DC | ANALYSIS_STATIC | CALC_RESIST_JACOBIAN |
                          CALC_RESIST_RESIDUAL | CALC_RESIST_LIM_RHS |
                          CALC_REACT_JACOBIAN | CALC_REACT_RESIDUAL |
                          CALC_REACT_LIM_RHS | CALC_OP | ENABLE_LIM | INIT_LIM;
    (void)inst.eval(model, sim, flags);
    sim.clear();
    inst.load_residuals(model, sim);
    inst.load_jacobian(model, sim);

    auto read_current = [&](const std::string &name, const std::vector<double> &residuals) -> double {
      auto it = sim.node_index.find(name);
      if (it == sim.node_index.end()) {
        return 0.0;
      }
      return -residuals[it->second];
    };

    double id_spice = read_current("d", sim.residual_resist);
    double ig_spice = read_current("g", sim.residual_resist);
    double is_spice = read_current("s", sim.residual_resist);
    double ie_spice = read_current("e", sim.residual_resist);

    std::cout << "Id=" << std::setprecision(12) << id_spice
              << " Ig=" << ig_spice
              << " Is=" << is_spice
              << " Ie=" << ie_spice << "\n";

    if (print_charges) {
      double qg = 0.0, qd = 0.0, qs = 0.0, qb = 0.0;
      bool qg_ok = read_opvar_value(desc, inst.data(), model.data(), "qg", &qg)
                   || read_opvar_value(desc, inst.data(), model.data(), "qgate", &qg);
      bool qd_ok = read_opvar_value(desc, inst.data(), model.data(), "qd", &qd)
                   || read_opvar_value(desc, inst.data(), model.data(), "qdrain", &qd);
      bool qs_ok = read_opvar_value(desc, inst.data(), model.data(), "qs", &qs)
                   || read_opvar_value(desc, inst.data(), model.data(), "qsource", &qs);
      bool qb_ok = read_opvar_value(desc, inst.data(), model.data(), "qb", &qb)
                   || read_opvar_value(desc, inst.data(), model.data(), "qbulk", &qb)
                   || read_opvar_value(desc, inst.data(), model.data(), "qe", &qb);
      if (!(qg_ok && qd_ok && qs_ok && qb_ok)) {
        std::cerr << "Warning: missing charge opvars\n";
      }
      std::cout << "Qg=" << qg << " Qd=" << qd << " Qs=" << qs << " Qb=" << qb << "\n";
    }

    if (print_derivs) {
      double gm = 0.0, gds = 0.0, gmb = 0.0;
      bool gm_ok = read_opvar_value(desc, inst.data(), model.data(), "gm", &gm);
      bool gds_ok = read_opvar_value(desc, inst.data(), model.data(), "gds", &gds);
      bool gmb_ok = read_opvar_value(desc, inst.data(), model.data(), "gmbs", &gmb)
                    || read_opvar_value(desc, inst.data(), model.data(), "gmb", &gmb);
      if (!(gm_ok && gds_ok && gmb_ok)) {
        std::cerr << "Warning: missing gm/gds/gmb opvars\n";
      }
      std::cout << "Gm=" << gm << " Gds=" << gds << " Gmb=" << gmb << "\n";
    }

    if (print_cap) {
      std::vector<double> g_full;
      std::vector<double> c_full;
      build_full_jacobian(sim, sim.jacobian_resist, g_full);
      build_full_jacobian(sim, sim.jacobian_react, c_full);
      std::vector<double> c_condensed;
      condense_capacitance(g_full, c_full, sim.node_names.size(),
                           sim.terminal_indices, sim.internal_indices, c_condensed);
      auto idx_of = [&](const std::string &name) -> int {
        for (std::size_t i = 0; i < sim.terminal_indices.size(); ++i) {
          if (sim.node_names[sim.terminal_indices[i]] == name) {
            return static_cast<int>(i);
          }
        }
        return -1;
      };
      double cgg = 0.0, cgd = 0.0, cgs = 0.0, cgb = 0.0;
      double cdg = 0.0, cdd = 0.0;
      int g = idx_of("g");
      int d = idx_of("d");
      int s = idx_of("s");
      int e = idx_of("e");
      if (g >= 0 && !c_condensed.empty()) {
        std::size_t n = sim.terminal_indices.size();
        cgg = c_condensed[g * n + g];
        if (d >= 0) cgd = c_condensed[g * n + d];
        if (s >= 0) cgs = c_condensed[g * n + s];
        if (e >= 0) cgb = c_condensed[g * n + e];
        if (d >= 0) {
          cdg = c_condensed[d * n + g];
          cdd = c_condensed[d * n + d];
        }
      }
      std::cout << "Cgg=" << cgg << " Cgd=" << cgd
                << " Cgs=" << cgs << " Cgb=" << cgb
                << " Cdg=" << cdg << " Cdd=" << cdd << "\n";
    }

    if (!quiet && !print_charges && !print_cap) {
      std::cout << "Terminal currents (A, from residuals):\n";
      std::cout << "  Id = " << std::setprecision(12) << id_spice << "\n";
      std::cout << "  Ig = " << std::setprecision(12) << ig_spice << "\n";
      std::cout << "  Is = " << std::setprecision(12) << is_spice << "\n";
      std::cout << "  Ie = " << std::setprecision(12) << ie_spice << "\n";

      if (!sim.internal_indices.empty()) {
        std::cout << "Internal nodes (voltage, residual):\n";
        for (auto idx : sim.internal_indices) {
          const std::string &name = sim.node_names[idx];
          std::cout << "  " << name << " V=" << std::setprecision(12) << sim.solve[idx]
                    << " res=" << std::setprecision(12) << sim.residual_resist[idx] << "\n";
        }
      }

      std::cout << "Residuals (resistive):\n";
      for (std::size_t i = 0; i < sim.node_names.size(); ++i) {
        std::cout << "  " << sim.node_names[i] << " = " << std::setprecision(12)
                  << sim.residual_resist[i] << "\n";
      }

      std::cout << "Jacobian (resistive):\n";
      for (std::size_t i = 0; i < sim.jacobian_info.size(); ++i) {
        auto row = sim.jacobian_info[i].first;
        auto col = sim.jacobian_info[i].second;
        const char *row_name = row < sim.node_names.size() ? sim.node_names[row].c_str() : "?";
        const char *col_name = col < sim.node_names.size() ? sim.node_names[col].c_str() : "?";
        std::cout << "  dI(" << row_name << ")/dV(" << col_name << ") = "
                  << std::setprecision(12) << sim.jacobian_resist[i] << "\n";
      }
    }
  } catch (const std::exception &exc) {
    std::cerr << "OSDI eval failed: " << exc.what() << "\n";
    return 1;
  }

  return 0;
}
