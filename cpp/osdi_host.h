#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "osdi_0_3.h"

namespace osdi_host {

class OsdiLibrary {
public:
  static OsdiLibrary load(const std::string &path);
  ~OsdiLibrary();

  OsdiLibrary(const OsdiLibrary &) = delete;
  OsdiLibrary &operator=(const OsdiLibrary &) = delete;
  OsdiLibrary(OsdiLibrary &&other) noexcept;
  OsdiLibrary &operator=(OsdiLibrary &&other) noexcept;

  const OsdiDescriptor *descriptor(std::size_t index = 0) const;
  const OsdiDescriptor *descriptor_by_name(const std::string &name) const;
  std::size_t descriptor_count() const;

private:
  explicit OsdiLibrary(void *handle, const OsdiDescriptor *descriptors, std::uint32_t count);
  void *handle_ = nullptr;
  const OsdiDescriptor *descriptors_ = nullptr;
  std::uint32_t num_descriptors_ = 0;
};

class OsdiModel {
public:
  explicit OsdiModel(const OsdiDescriptor *descriptor);
  ~OsdiModel();

  OsdiModel(const OsdiModel &) = delete;
  OsdiModel &operator=(const OsdiModel &) = delete;

  void process_params();
  void set_param(const std::string &name, double value);
  const OsdiDescriptor *descriptor() const { return descriptor_; }
  void *data() const { return model_data_; }

private:
  const OsdiDescriptor *descriptor_ = nullptr;
  void *model_data_ = nullptr;
};

struct OsdiSimulation {
  std::vector<std::string> node_names;
  std::unordered_map<std::string, std::uint32_t> node_index;
  std::vector<std::uint32_t> terminal_indices;
  std::vector<std::uint32_t> internal_indices;
  std::vector<double> residual_resist;
  std::vector<double> residual_react;
  std::vector<double> rhs_tran;
  std::vector<double> solve;
  std::vector<double> prev_solve;
  bool has_prev_solve = false;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> jacobian_info;
  std::unordered_map<std::uint64_t, std::size_t> jacobian_index;
  std::vector<double> jacobian_resist;
  std::vector<double> jacobian_react;
  std::vector<double> state_prev;
  std::vector<double> state_next;
  std::vector<double> noise_dense;
  std::vector<const char *> sim_param_names;
  std::vector<double> sim_param_vals;
  std::vector<const char *> sim_param_names_str;
  std::vector<const char *> sim_param_vals_str;
  std::vector<std::string> sim_param_storage;

  OsdiSimulation();
  std::uint32_t register_node(const std::string &name);
  void register_jacobian_entry(std::uint32_t row, std::uint32_t col);
  std::size_t get_jacobian_entry(std::uint32_t row, std::uint32_t col) const;
  void build_jacobian();
  void clear();
  void set_voltage(const std::string &node, double voltage);
  void set_sim_param(const std::string &name, double value);
};

class OsdiInstance {
public:
  explicit OsdiInstance(const OsdiDescriptor *descriptor);
  ~OsdiInstance();

  OsdiInstance(const OsdiInstance &) = delete;
  OsdiInstance &operator=(const OsdiInstance &) = delete;

  std::vector<std::uint32_t> process_params(
      const OsdiModel &model,
      std::uint32_t connected_terminals,
      double temperature);

  void bind_simulation(
      OsdiSimulation &sim,
      const OsdiModel &model,
      std::uint32_t connected_terminals,
      double temperature);

  std::uint32_t eval(
      const OsdiModel &model,
      OsdiSimulation &sim,
      std::uint32_t flags);
  std::uint32_t eval_with_time(
      const OsdiModel &model,
      OsdiSimulation &sim,
      std::uint32_t flags,
      double abstime);

  void load_residuals(const OsdiModel &model, OsdiSimulation &sim);
  void load_jacobian(const OsdiModel &model, OsdiSimulation &sim);
  void load_spice_rhs_dc(const OsdiModel &model, OsdiSimulation &sim);
  void load_spice_rhs_tran(const OsdiModel &model, OsdiSimulation &sim, double alpha);
  void load_jacobian_tran(const OsdiModel &model, OsdiSimulation &sim, double alpha);
  bool solve_internal_nodes(
      const OsdiModel &model,
      OsdiSimulation &sim,
      int max_iter,
      double tol);
  bool solve_internal_nodes_tran(
      const OsdiModel &model,
      OsdiSimulation &sim,
      double abstime,
      double alpha,
      int max_iter,
      double tol);

  const OsdiDescriptor *descriptor() const { return descriptor_; }
  void *data() const { return inst_data_; }

private:
  std::vector<std::uint32_t> collapse_nodes(std::uint32_t connected_terminals);

  const OsdiDescriptor *descriptor_ = nullptr;
  void *inst_data_ = nullptr;
};

}  // namespace osdi_host
