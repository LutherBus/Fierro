// Luther - Added a laser structure that contains the laser path as well as the parameters defining its shape
// There are currently two laser shapes: Goldak and Spherical

#pragma once

#include <string>
#include "additive_data.hpp" 

enum class LaserModelType
{
    SPHERICAL,
    GOLDAK
};

struct Laser_t
{
    // --- Tabular path data ---
    std::string  path_file;        // path to the laser scan-path file
    ToolPathInfo tool_path_info;   // parsed (x,y,z,power,state...) points, sized by load_laser_path
    
    LaserModelType model;
    
    double radius = 0.0;

    double absorptivity = 0.0;
    double major_front = 0.0;
    double major_rear = 0.0;
    double minor = 0.0;
    double depth = 0.0;

    Laser_t() = default;
};

