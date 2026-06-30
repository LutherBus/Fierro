#pragma once

#include <string>           // for std::string
#include "additive_data.hpp"   // or wherever LaserPathTable is declared — see below

struct Laser_t
{
    // --- Tabular path data ---
    std::string  path_file;        // path to the laser scan-path file
    ToolPathInfo tool_path_info;   // parsed (x,y,z,power,state...) points, sized by load_laser_path
    Laser_t() = default;
};

extern std::vector<std::string> str_laser_path_inps;
extern std::vector<std::string> laser_path_required_inps;