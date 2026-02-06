#include "osdi_host.h"

#include <iostream>
#include <string>

namespace {

void print_usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [--osdi PATH] [--list] [--describe]\n";
}

}  // namespace

int main(int argc, char **argv) {
  std::string osdi_path = "build/osdi/bsimcmg.osdi";
  bool list_only = false;
  bool describe = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--osdi" && i + 1 < argc) {
      osdi_path = argv[++i];
    } else if (arg == "--list") {
      list_only = true;
    } else if (arg == "--describe") {
      describe = true;
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
    std::size_t count = lib.descriptor_count();
    std::cout << "OSDI descriptors: " << count << "\n";
    for (std::size_t i = 0; i < count; ++i) {
      const OsdiDescriptor *desc = lib.descriptor(i);
      if (!desc) {
        continue;
      }
      std::cout << "- [" << i << "] " << (desc->name ? desc->name : "(null)") << "\n";
      if (describe) {
        std::cout << "  nodes=" << desc->num_nodes << " terminals=" << desc->num_terminals
                  << " params=" << desc->num_params << " inst_params=" << desc->num_instance_params
                  << " opvars=" << desc->num_opvars << "\n";
      }
    }
    if (list_only || describe) {
      return 0;
    }
  } catch (const std::exception &exc) {
    std::cerr << "Failed to load OSDI: " << exc.what() << "\n";
    return 1;
  }

  return 0;
}
