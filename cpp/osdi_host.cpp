#include "osdi_host.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include <dlfcn.h>
#include <stdlib.h>

namespace osdi_host {
namespace {

constexpr const char *kOsdiInstanceName = "osdi_host";

using OsdiLogPtr = void (*)(void *handle, const char *msg, std::uint32_t lvl);

std::string dl_error_string() {
  const char *err = dlerror();
  return err ? std::string(err) : std::string("unknown dlerror");
}

void *aligned_alloc_zero(std::size_t size) {
  if (size == 0) {
    return nullptr;
  }
  std::size_t alignment = alignof(max_align_t);
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, ((size + alignment - 1) / alignment) * alignment) != 0) {
    return nullptr;
  }
  std::memset(ptr, 0, size);
  return ptr;
}

void aligned_free(void *ptr) {
  free(ptr);
}

void check_init_result(const OsdiDescriptor *descriptor, OsdiInitInfo &info) {
  auto cleanup = [&info]() {
    if (info.num_errors != 0 && info.errors != nullptr) {
      free(info.errors);
      info.errors = nullptr;
      info.num_errors = 0;
    }
  };

  if ((info.flags & EVAL_RET_FLAG_FATAL) != 0) {
    cleanup();
    throw std::runtime_error("OSDI fatal error reported during setup");
  }

  if (info.num_errors == 0) {
    cleanup();
    return;
  }

  std::string message;
  bool fatal = false;
  for (std::uint32_t i = 0; i < info.num_errors; ++i) {
    const OsdiInitError &err = info.errors[i];
    if (err.code == INIT_ERR_OUT_OF_BOUNDS) {
      std::uint32_t param_id = err.payload.parameter_id;
      if (descriptor && descriptor->param_opvar && param_id < descriptor->num_params) {
        const OsdiParamOpvar &param = descriptor->param_opvar[param_id];
        const char *name = param.name ? param.name[0] : "unknown";
        message.append("parameter out of bounds: ");
        message.append(name ? name : "unknown");
        message.push_back('\n');
      } else {
        message.append("parameter out of bounds: unknown\n");
      }
    } else {
      fatal = true;
      message.append("unknown OSDI init error\n");
    }
  }
  cleanup();
  if (!message.empty()) {
    if (message.back() == '\n') {
      message.pop_back();
    }
    if (fatal) {
      throw std::runtime_error(message);
    }
    std::cerr << "OSDI init warning: " << message << "\n";
  }
}

void osdi_log_handler(void *handle, const char *msg, std::uint32_t lvl) {
  const char *instance = handle ? static_cast<const char *>(handle) : "osdi";
  const char *text = msg ? msg : "";
  std::cerr << "osdi[" << instance << "] lvl=" << (lvl & LOG_LVL_MASK) << " " << text << "\n";
}

extern "C" double osdi_pnjlim(
    bool init,
    bool *check,
    double vnew,
    double vold,
    double vt,
    double vcrit) {
  bool triggered = false;
  if (init) {
    vnew = vcrit;
    triggered = true;
  } else if (vnew > vcrit && std::abs(vnew - vold) > 2.0 * vt) {
    if (vold > 0.0) {
      double arg = (vnew - vold) / vt;
      if (arg > 0.0) {
        vnew = vold + vt * std::log(arg + 1.0);
      } else {
        vnew = vcrit;
      }
    } else {
      vnew = vt * std::log(vnew / vt);
    }
    triggered = true;
  }
  if (check) {
    *check = triggered;
  }
  return vnew;
}

void install_lim_functions(void *handle) {
  auto table_sym = reinterpret_cast<OsdiLimFunction **>(dlsym(handle, "OSDI_LIM_TABLE"));
  auto len_sym = reinterpret_cast<std::uint32_t *>(dlsym(handle, "OSDI_LIM_TABLE_LEN"));
  if (!table_sym || !len_sym || !*table_sym) {
    return;
  }

  auto lim_table = *table_sym;
  auto len = *len_sym;
  for (std::uint32_t i = 0; i < len; ++i) {
    OsdiLimFunction &entry = lim_table[i];
    if (entry.name && std::strcmp(entry.name, "pnjlim") == 0) {
      entry.func_ptr = reinterpret_cast<void *>(&osdi_pnjlim);
    }
  }
}

}  // namespace

OsdiLibrary::OsdiLibrary(void *handle, const OsdiDescriptor *descriptors, std::uint32_t count)
    : handle_(handle), descriptors_(descriptors), num_descriptors_(count) {}

OsdiLibrary::~OsdiLibrary() {
  if (handle_) {
    dlclose(handle_);
  }
}

OsdiLibrary::OsdiLibrary(OsdiLibrary &&other) noexcept
    : handle_(other.handle_), descriptors_(other.descriptors_), num_descriptors_(other.num_descriptors_) {
  other.handle_ = nullptr;
  other.descriptors_ = nullptr;
  other.num_descriptors_ = 0;
}

OsdiLibrary &OsdiLibrary::operator=(OsdiLibrary &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (handle_) {
    dlclose(handle_);
  }
  handle_ = other.handle_;
  descriptors_ = other.descriptors_;
  num_descriptors_ = other.num_descriptors_;
  other.handle_ = nullptr;
  other.descriptors_ = nullptr;
  other.num_descriptors_ = 0;
  return *this;
}

OsdiLibrary OsdiLibrary::load(const std::string &path) {
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    throw std::runtime_error("dlopen failed: " + dl_error_string());
  }

  auto major_sym = reinterpret_cast<std::uint32_t *>(dlsym(handle, "OSDI_VERSION_MAJOR"));
  auto minor_sym = reinterpret_cast<std::uint32_t *>(dlsym(handle, "OSDI_VERSION_MINOR"));
  if (!major_sym || !minor_sym) {
    dlclose(handle);
    throw std::runtime_error("missing OSDI version symbols");
  }
  if (*major_sym != 0 || *minor_sym != 3) {
    dlclose(handle);
    throw std::runtime_error("unsupported OSDI version");
  }

  auto count_sym = reinterpret_cast<std::uint32_t *>(dlsym(handle, "OSDI_NUM_DESCRIPTORS"));
  auto desc_sym = reinterpret_cast<OsdiDescriptor *>(dlsym(handle, "OSDI_DESCRIPTORS"));
  if (!count_sym || !desc_sym) {
    dlclose(handle);
    throw std::runtime_error("missing OSDI descriptor symbols");
  }

  if (auto log_sym = reinterpret_cast<OsdiLogPtr *>(dlsym(handle, "osdi_log"))) {
    *log_sym = &osdi_log_handler;
  }

  install_lim_functions(handle);

  return OsdiLibrary(handle, desc_sym, *count_sym);
}

const OsdiDescriptor *OsdiLibrary::descriptor(std::size_t index) const {
  if (index >= num_descriptors_) {
    return nullptr;
  }
  return &descriptors_[index];
}

const OsdiDescriptor *OsdiLibrary::descriptor_by_name(const std::string &name) const {
  for (std::uint32_t i = 0; i < num_descriptors_; ++i) {
    const OsdiDescriptor &desc = descriptors_[i];
    if (desc.name && name == desc.name) {
      return &desc;
    }
  }
  return nullptr;
}

std::size_t OsdiLibrary::descriptor_count() const {
  return num_descriptors_;
}

OsdiModel::OsdiModel(const OsdiDescriptor *descriptor) : descriptor_(descriptor) {
  if (!descriptor_) {
    throw std::runtime_error("descriptor is null");
  }
  model_data_ = aligned_alloc_zero(descriptor_->model_size);
  if (descriptor_->model_size != 0 && !model_data_) {
    throw std::runtime_error("failed to allocate model data");
  }
}

OsdiModel::~OsdiModel() {
  aligned_free(model_data_);
}

void OsdiModel::process_params() {
  OsdiSimParas sim_params{};
  OsdiInitInfo info{};
  descriptor_->setup_model(const_cast<char *>(kOsdiInstanceName), model_data_, &sim_params, &info);
  check_init_result(descriptor_, info);
}

void OsdiModel::set_param(const std::string &name, double value) {
  if (!descriptor_ || !descriptor_->param_opvar) {
    throw std::runtime_error("descriptor has no parameter metadata");
  }
  for (std::uint32_t i = 0; i < descriptor_->num_params; ++i) {
    const OsdiParamOpvar &param = descriptor_->param_opvar[i];
    const char *param_name = param.name ? param.name[0] : nullptr;
    if (param_name && name == param_name) {
      void *ptr = descriptor_->access(nullptr, model_data_, i, ACCESS_FLAG_SET);
      if (!ptr) {
        throw std::runtime_error("invalid parameter access");
      }
      auto *value_ptr = reinterpret_cast<double *>(ptr);
      *value_ptr = value;
      return;
    }
  }
  throw std::runtime_error("parameter not found: " + name);
}

OsdiSimulation::OsdiSimulation() {
  node_names.emplace_back("gnd");
  node_index.emplace("gnd", 0);
  residual_resist.push_back(0.0);
  residual_react.push_back(0.0);
  rhs_tran.push_back(0.0);
  solve.push_back(0.0);
  prev_solve.push_back(0.0);
  jacobian_info.emplace_back(0, 0);
  jacobian_index.emplace(0, 0);
  sim_param_storage.emplace_back();
  sim_param_names.push_back(nullptr);
  sim_param_names_str.push_back(nullptr);
  sim_param_vals.push_back(0.0);
}

std::uint32_t OsdiSimulation::register_node(const std::string &name) {
  auto it = node_index.find(name);
  if (it != node_index.end()) {
    return it->second;
  }
  std::uint32_t idx = static_cast<std::uint32_t>(node_names.size());
  node_names.push_back(name);
  node_index.emplace(name, idx);
  residual_resist.push_back(0.0);
  residual_react.push_back(0.0);
  rhs_tran.push_back(0.0);
  solve.push_back(0.0);
  prev_solve.push_back(0.0);
  return idx;
}

void OsdiSimulation::register_jacobian_entry(std::uint32_t row, std::uint32_t col) {
  if (row == 0 || col == 0) {
    return;
  }
  std::uint64_t key = (static_cast<std::uint64_t>(row) << 32) | col;
  if (jacobian_index.find(key) != jacobian_index.end()) {
    return;
  }
  std::size_t idx = jacobian_info.size();
  jacobian_info.emplace_back(row, col);
  jacobian_index.emplace(key, idx);
}

std::size_t OsdiSimulation::get_jacobian_entry(std::uint32_t row, std::uint32_t col) const {
  if (row == 0 || col == 0) {
    return 0;
  }
  std::uint64_t key = (static_cast<std::uint64_t>(row) << 32) | col;
  auto it = jacobian_index.find(key);
  if (it == jacobian_index.end()) {
    return 0;
  }
  return it->second;
}

void OsdiSimulation::build_jacobian() {
  jacobian_resist.assign(jacobian_info.size(), 0.0);
  jacobian_react.assign(jacobian_info.size(), 0.0);
}

void OsdiSimulation::clear() {
  std::fill(residual_resist.begin(), residual_resist.end(), 0.0);
  std::fill(residual_react.begin(), residual_react.end(), 0.0);
  std::fill(rhs_tran.begin(), rhs_tran.end(), 0.0);
  std::fill(jacobian_resist.begin(), jacobian_resist.end(), 0.0);
  std::fill(jacobian_react.begin(), jacobian_react.end(), 0.0);
}

void OsdiSimulation::set_voltage(const std::string &node, double voltage) {
  auto it = node_index.find(node);
  if (it == node_index.end()) {
    throw std::runtime_error("unknown node: " + node);
  }
  solve[it->second] = voltage;
}

void OsdiSimulation::set_sim_param(const std::string &name, double value) {
  for (std::size_t i = 0; i < sim_param_names.size(); ++i) {
    if (sim_param_names[i] == nullptr) {
      break;
    }
    if (name == sim_param_names[i]) {
      sim_param_vals[i] = value;
      return;
    }
  }
  std::size_t insert_pos = 0;
  while (insert_pos < sim_param_names.size() && sim_param_names[insert_pos] != nullptr) {
    ++insert_pos;
  }
  sim_param_storage.push_back(name);
  if (insert_pos >= sim_param_names.size()) {
    sim_param_names.push_back(sim_param_storage.back().c_str());
    sim_param_vals.push_back(value);
    sim_param_names.push_back(nullptr);
    sim_param_vals.push_back(0.0);
  } else {
    sim_param_names.insert(sim_param_names.begin() + static_cast<std::ptrdiff_t>(insert_pos),
                           sim_param_storage.back().c_str());
    sim_param_vals.insert(sim_param_vals.begin() + static_cast<std::ptrdiff_t>(insert_pos), value);
  }
}

OsdiInstance::OsdiInstance(const OsdiDescriptor *descriptor) : descriptor_(descriptor) {
  if (!descriptor_) {
    throw std::runtime_error("descriptor is null");
  }
  inst_data_ = aligned_alloc_zero(descriptor_->instance_size);
  if (descriptor_->instance_size != 0 && !inst_data_) {
    throw std::runtime_error("failed to allocate instance data");
  }
}

OsdiInstance::~OsdiInstance() {
  aligned_free(inst_data_);
}

std::vector<std::uint32_t> OsdiInstance::process_params(
    const OsdiModel &model,
    std::uint32_t connected_terminals,
    double temperature) {
  OsdiSimParas sim_params{};
  OsdiInitInfo info{};
  descriptor_->setup_instance(
      const_cast<char *>(kOsdiInstanceName),
      inst_data_,
      model.data(),
      temperature,
      connected_terminals,
      &sim_params,
      &info);
  check_init_result(descriptor_, info);
  return collapse_nodes(connected_terminals);
}

void OsdiInstance::bind_simulation(
    OsdiSimulation &sim,
    const OsdiModel &model,
    std::uint32_t connected_terminals,
    double temperature) {
  std::vector<std::uint32_t> internal_nodes =
      process_params(model, connected_terminals, temperature);

  const OsdiNode *nodes = descriptor_->nodes;
  std::vector<std::uint32_t> terminal_indices;
  terminal_indices.reserve(connected_terminals);
  for (std::uint32_t i = 0; i < connected_terminals; ++i) {
    terminal_indices.push_back(sim.register_node(nodes[i].name ? nodes[i].name : ""));
  }

  sim.terminal_indices = terminal_indices;

  std::vector<std::uint32_t> internal_indices;
  for (auto &node_idx : internal_nodes) {
    const OsdiNode &node = nodes[node_idx];
    node_idx = sim.register_node(node.name ? node.name : "");
    internal_indices.push_back(node_idx);
  }
  sim.internal_indices = internal_indices;

  auto *mapping = reinterpret_cast<std::uint32_t *>(
      static_cast<std::uint8_t *>(inst_data_) + descriptor_->node_mapping_offset);
  for (std::uint32_t i = 0; i < descriptor_->num_nodes; ++i) {
    std::uint32_t idx = mapping[i];
    if (idx < terminal_indices.size()) {
      mapping[i] = terminal_indices[idx];
    } else if (idx == UINT32_MAX) {
      mapping[i] = 0;
    } else {
      std::size_t internal_idx = idx - terminal_indices.size();
      if (internal_idx >= internal_nodes.size()) {
        mapping[i] = 0;
      } else {
        mapping[i] = internal_nodes[internal_idx];
      }
    }
  }

  for (std::uint32_t i = 0; i < descriptor_->num_jacobian_entries; ++i) {
    const OsdiJacobianEntry &entry = descriptor_->jacobian_entries[i];
    std::uint32_t row = mapping[entry.nodes.node_1];
    std::uint32_t column = mapping[entry.nodes.node_2];
    sim.register_jacobian_entry(row, column);
  }
  sim.build_jacobian();

  auto **ptr_resist = reinterpret_cast<double **>(
      static_cast<std::uint8_t *>(inst_data_) + descriptor_->jacobian_ptr_resist_offset);
  for (std::uint32_t i = 0; i < descriptor_->num_jacobian_entries; ++i) {
    const OsdiJacobianEntry &entry = descriptor_->jacobian_entries[i];
    std::uint32_t row = mapping[entry.nodes.node_1];
    std::uint32_t column = mapping[entry.nodes.node_2];
    std::size_t idx = sim.get_jacobian_entry(row, column);
    ptr_resist[i] = sim.jacobian_resist.data() + idx;
    if (entry.react_ptr_off != UINT32_MAX) {
      auto **react_ptr =
          reinterpret_cast<double **>(static_cast<std::uint8_t *>(inst_data_) + entry.react_ptr_off);
      *react_ptr = sim.jacobian_react.data() + idx;
    }
  }

  sim.state_prev.assign(descriptor_->num_states, 0.0);
  sim.state_next.assign(descriptor_->num_states, 0.0);
  sim.noise_dense.assign(descriptor_->num_noise_src, 0.0);
}

std::uint32_t OsdiInstance::eval(
    const OsdiModel &model,
    OsdiSimulation &sim,
    std::uint32_t flags) {
  return eval_with_time(model, sim, flags, 0.0);
}

std::uint32_t OsdiInstance::eval_with_time(
    const OsdiModel &model,
    OsdiSimulation &sim,
    std::uint32_t flags,
    double abstime) {
  OsdiSimParas sim_params{};
  sim_params.names = const_cast<char **>(sim.sim_param_names.data());
  sim_params.vals = sim.sim_param_vals.data();
  sim_params.names_str = const_cast<char **>(sim.sim_param_names_str.data());
  sim_params.vals_str = const_cast<char **>(sim.sim_param_vals_str.data());

  OsdiSimInfo sim_info{};
  sim_info.paras = sim_params;
  sim_info.abstime = abstime;
  if (sim.has_prev_solve) {
    sim_info.prev_solve = sim.prev_solve.data();
  } else {
    sim_info.prev_solve = sim.solve.data();
  }
  sim_info.prev_state = sim.state_prev.data();
  sim_info.next_state = sim.state_next.data();
  sim_info.flags = flags;

  return descriptor_->eval(
      const_cast<char *>(kOsdiInstanceName), inst_data_, model.data(), &sim_info);
}

void OsdiInstance::load_residuals(const OsdiModel &model, OsdiSimulation &sim) {
  descriptor_->load_residual_resist(inst_data_, model.data(), sim.residual_resist.data());
  descriptor_->load_limit_rhs_resist(inst_data_, model.data(), sim.residual_resist.data());
  descriptor_->load_residual_react(inst_data_, model.data(), sim.residual_react.data());
  descriptor_->load_limit_rhs_react(inst_data_, model.data(), sim.residual_react.data());
}

void OsdiInstance::load_jacobian(const OsdiModel &model, OsdiSimulation &sim) {
  (void)sim;
  descriptor_->load_jacobian_resist(inst_data_, model.data());
  descriptor_->load_jacobian_react(inst_data_, model.data(), 1.0);
}

void OsdiInstance::load_spice_rhs_dc(const OsdiModel &model, OsdiSimulation &sim) {
  descriptor_->load_spice_rhs_dc(
      inst_data_, model.data(), sim.residual_resist.data(), sim.solve.data());
}

void OsdiInstance::load_spice_rhs_tran(
    const OsdiModel &model,
    OsdiSimulation &sim,
    double alpha) {
  descriptor_->load_spice_rhs_tran(
      inst_data_, model.data(), sim.rhs_tran.data(), sim.prev_solve.data(), alpha);
}

void OsdiInstance::load_jacobian_tran(
    const OsdiModel &model,
    OsdiSimulation &sim,
    double alpha) {
  (void)sim;
  descriptor_->load_jacobian_tran(inst_data_, model.data(), alpha);
}

namespace {

bool solve_linear_system(std::vector<double> &a, std::vector<double> &b, int n) {
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
      std::swap(b[i], b[pivot]);
    }
    double diag = a[i * n + i];
    for (int c = i; c < n; ++c) {
      a[i * n + c] /= diag;
    }
    b[i] /= diag;
    for (int r = i + 1; r < n; ++r) {
      double factor = a[r * n + i];
      if (factor == 0.0) {
        continue;
      }
      for (int c = i; c < n; ++c) {
        a[r * n + c] -= factor * a[i * n + c];
      }
      b[r] -= factor * b[i];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    for (int r = 0; r < i; ++r) {
      double factor = a[r * n + i];
      if (factor == 0.0) {
        continue;
      }
      a[r * n + i] = 0.0;
      b[r] -= factor * b[i];
    }
  }
  return true;
}

}  // namespace

bool OsdiInstance::solve_internal_nodes(
    const OsdiModel &model,
    OsdiSimulation &sim,
    int max_iter,
    double tol) {
  if (sim.internal_indices.empty()) {
    return true;
  }

  std::unordered_map<std::uint32_t, int> internal_pos;
  internal_pos.reserve(sim.internal_indices.size());
  for (std::size_t i = 0; i < sim.internal_indices.size(); ++i) {
    internal_pos.emplace(sim.internal_indices[i], static_cast<int>(i));
  }

  auto residual_norm = [&](const std::vector<double> &res) {
    double max_abs = 0.0;
    for (auto idx : sim.internal_indices) {
      max_abs = std::max(max_abs, std::abs(res[idx]));
    }
    return max_abs;
  };

  std::vector<double> base_solve = sim.solve;

  for (int iter = 0; iter < max_iter; ++iter) {
    std::uint32_t flags = ANALYSIS_DC | ANALYSIS_STATIC | CALC_RESIST_RESIDUAL |
                          CALC_RESIST_JACOBIAN | CALC_RESIST_LIM_RHS |
                          ENABLE_LIM | INIT_LIM;
    eval(model, sim, flags);
    sim.clear();
    load_residuals(model, sim);
    load_jacobian(model, sim);

    double norm = residual_norm(sim.residual_resist);
    if (norm < tol) {
      return true;
    }

    int n = static_cast<int>(sim.internal_indices.size());
    std::vector<double> a(n * n, 0.0);
    std::vector<double> b(n, 0.0);
    for (int i = 0; i < n; ++i) {
      b[i] = -sim.residual_resist[sim.internal_indices[i]];
    }

    for (std::size_t k = 0; k < sim.jacobian_info.size(); ++k) {
      auto row = sim.jacobian_info[k].first;
      auto col = sim.jacobian_info[k].second;
      auto it_r = internal_pos.find(row);
      auto it_c = internal_pos.find(col);
      if (it_r == internal_pos.end() || it_c == internal_pos.end()) {
        continue;
      }
      int r = it_r->second;
      int c = it_c->second;
      a[r * n + c] = sim.jacobian_resist[k];
    }

    std::vector<double> a_copy = a;
    std::vector<double> delta = b;
    if (!solve_linear_system(a_copy, delta, n)) {
      return false;
    }

    for (int i = 0; i < n; ++i) {
      auto idx = sim.internal_indices[i];
      double update = delta[i];
      if (update > 0.2) {
        update = 0.2;
      } else if (update < -0.2) {
        update = -0.2;
      }
      sim.solve[idx] += update;
    }
  }
  return false;
}

bool OsdiInstance::solve_internal_nodes_tran(
    const OsdiModel &model,
    OsdiSimulation &sim,
    double abstime,
    double alpha,
    int max_iter,
    double tol) {
  if (sim.internal_indices.empty()) {
    return true;
  }

  std::unordered_map<std::uint32_t, int> internal_pos;
  internal_pos.reserve(sim.internal_indices.size());
  for (std::size_t i = 0; i < sim.internal_indices.size(); ++i) {
    internal_pos.emplace(sim.internal_indices[i], static_cast<int>(i));
  }

  auto residual_norm = [&](const std::vector<double> &res) {
    double max_abs = 0.0;
    for (auto idx : sim.internal_indices) {
      max_abs = std::max(max_abs, std::abs(res[idx]));
    }
    return max_abs;
  };

  for (int iter = 0; iter < max_iter; ++iter) {
    std::uint32_t flags = ANALYSIS_TRAN | CALC_RESIST_RESIDUAL | CALC_RESIST_JACOBIAN |
                          CALC_RESIST_LIM_RHS | CALC_REACT_RESIDUAL |
                          CALC_REACT_JACOBIAN | CALC_REACT_LIM_RHS |
                          ENABLE_LIM | INIT_LIM;
    eval_with_time(model, sim, flags, abstime);
    sim.clear();
    load_residuals(model, sim);
    load_jacobian_tran(model, sim, alpha);
    load_spice_rhs_tran(model, sim, alpha);

    std::vector<double> total_residual(sim.residual_resist.size(), 0.0);
    for (std::size_t i = 0; i < total_residual.size(); ++i) {
      total_residual[i] = sim.residual_resist[i] + alpha * sim.residual_react[i] - sim.rhs_tran[i];
    }

    double norm = residual_norm(total_residual);
    if (norm < tol) {
      return true;
    }

    int n = static_cast<int>(sim.internal_indices.size());
    std::vector<double> a(n * n, 0.0);
    std::vector<double> b(n, 0.0);
    for (int i = 0; i < n; ++i) {
      b[i] = -total_residual[sim.internal_indices[i]];
    }

    for (std::size_t k = 0; k < sim.jacobian_info.size(); ++k) {
      auto row = sim.jacobian_info[k].first;
      auto col = sim.jacobian_info[k].second;
      auto it_r = internal_pos.find(row);
      auto it_c = internal_pos.find(col);
      if (it_r == internal_pos.end() || it_c == internal_pos.end()) {
        continue;
      }
      int r = it_r->second;
      int c = it_c->second;
      a[r * n + c] = sim.jacobian_resist[k];
    }

    std::vector<double> a_copy = a;
    std::vector<double> delta = b;
    if (!solve_linear_system(a_copy, delta, n)) {
      return false;
    }

    for (int i = 0; i < n; ++i) {
      auto idx = sim.internal_indices[i];
      double update = delta[i];
      if (update > 0.2) {
        update = 0.2;
      } else if (update < -0.2) {
        update = -0.2;
      }
      sim.solve[idx] += update;
    }
  }
  return false;
}

std::vector<std::uint32_t> OsdiInstance::collapse_nodes(std::uint32_t connected_terminals) {
  std::vector<std::uint32_t> back_map;
  back_map.reserve(descriptor_->num_nodes - connected_terminals);
  for (std::uint32_t i = connected_terminals; i < descriptor_->num_nodes; ++i) {
    back_map.push_back(i);
  }

  auto *mapping = reinterpret_cast<std::uint32_t *>(
      static_cast<std::uint8_t *>(inst_data_) + descriptor_->node_mapping_offset);
  for (std::uint32_t i = 0; i < descriptor_->num_nodes; ++i) {
    mapping[i] = i;
  }

  auto *collapsed = reinterpret_cast<bool *>(
      static_cast<std::uint8_t *>(inst_data_) + descriptor_->collapsed_offset);

  for (std::uint32_t i = 0; i < descriptor_->num_collapsible; ++i) {
    if (!collapsed[i]) {
      continue;
    }
    OsdiNodePair candidate = descriptor_->collapsible[i];
    std::uint32_t from = candidate.node_1;
    std::uint32_t to = candidate.node_2;

    std::uint32_t mapped_from = mapping[from];
    bool collapse_to_gnd = (to == UINT32_MAX);
    std::uint32_t mapped_to = collapse_to_gnd ? UINT32_MAX : mapping[to];
    if (!collapse_to_gnd && mapped_to == UINT32_MAX) {
      collapse_to_gnd = true;
    }

    if (mapped_from < connected_terminals &&
        (collapse_to_gnd || mapped_to < connected_terminals)) {
      continue;
    }

    if (!collapse_to_gnd && mapped_from < mapped_to) {
      std::swap(mapped_from, mapped_to);
    }

    for (std::uint32_t j = 0; j < descriptor_->num_nodes; ++j) {
      std::uint32_t mapping_val = mapping[j];
      if (mapping_val == mapped_from) {
        mapping[j] = mapped_to;
      } else if (mapping_val > mapped_from && mapping_val != UINT32_MAX) {
        mapping[j] = mapping_val - 1;
      }
    }

    if (mapped_from >= connected_terminals) {
      std::size_t remove_idx = mapped_from - connected_terminals;
      if (remove_idx < back_map.size()) {
        back_map.erase(back_map.begin() + static_cast<std::ptrdiff_t>(remove_idx));
      }
    }
  }
  return back_map;
}

}  // namespace osdi_host
