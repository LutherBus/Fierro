/**********************************************************************************************
� 2020. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and
to permit others to do so.
This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
1.  Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.
3.  Neither the name of the copyright holder nor the names of its contributors may be used
to endorse or promote products derived from this software without specific prior
written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************/
#ifndef FIERRO_ADDITIVE_DATA_H
#define FIERRO_ADDITIVE_DATA_H

#include <stdio.h>
#include "matar.h"
#include "table.hpp"

// Luther - included additional libraries
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>


/////////////////////////////////////////////////////////////////////////////
///
/// \class ToolPathInfo
///
/// \brief Stores and manages additive manufacturing tool paths with time-parameterized 3D points
///        and provides utilities for obtaining the current tool position.
///
/// ToolPathInfo uses MATAR data types (e.g., CArray) and Table_t to represent
/// a sequence of tool path points, each defined by time, x, y, z, and power.
/// Provides methods for setting data points, updating device views, and interpolating
/// tool position and power at arbitrary times along the path.
///
/////////////////////////////////////////////////////////////////////////////
class ToolPathInfo {
public:
    
    enum Fields
    {
        time = 0,   ///<  time
        x = 1,      ///<  x position
        y = 2,      ///<  y position
        z = 3,      ///<  z position
        power = 4,  ///<  power
    };
    Table_t tool_path_table;

    size_t num_columns = 5;
    
    // Luther - added default constructor with no inputs
    // Default constructor
    ToolPathInfo() = default;

    // Default constructor that takes the number of data points
    ToolPathInfo(size_t npoints)
    {
        tool_path_table = Table_t(npoints, num_columns, "ToolPathInfo_table");
    }

    // Set a data point (time, x, y, z, power) at index i
    void set_data_point(size_t i, double time, double x, double y, double z, double power) {
        tool_path_table.set_value(i, Fields::time, time);
        tool_path_table.set_value(i, Fields::x, x);
        tool_path_table.set_value(i, Fields::y, y);
        tool_path_table.set_value(i, Fields::z, z);
        tool_path_table.set_value(i, Fields::power, power);
    }

    // Luther - added functionality to set a data point (time, x, y, z, power) by reading a file
    static ToolPathInfo load_laser_path(const std::string& filename) {
    // Pass 1: just count non-empty lines so we know how big to make the table.
    size_t npoints = 0;
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open laser path file: " + filename);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) npoints++;
        }
    }

    ToolPathInfo info(npoints);   // table is now allocated to the right size

    // Pass 2: re-open and actually parse into the table.
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open laser path file: " + filename);
    }
    std::string line;
    size_t i = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string field;
        std::vector<double> vals;
        while (std::getline(ss, field, ',')) {
            vals.push_back(std::stod(field));
        }
       
        if (vals.size() != 5) {
            throw std::runtime_error("Malformed line " + std::to_string(i) +
                " in " + filename + " (expected 5 fields, got " +
                std::to_string(vals.size()) + ")");
        }
        // vals[0] is the file's own index column — ignored, same as before.
        info.set_data_point(i, vals[0], vals[1], vals[2], vals[3], vals[4]);
        i++;
    }
    
    std::cout << "additive_data.hpp?" << std::endl;
    return info;
}
    
    // Update the device views (copy to the GPU from the CPU)
    void update_device() {
        tool_path_table.update_device();
    }

    // Compute current position of tool at time t on the device, assuming linear motion between path points.
    // Returns a std::array<double,3> {x, y, z}
    KOKKOS_INLINE_FUNCTION
    void get_position(const double& t, double& x, double& y, double& z) const {

        // get the x position at time t
        x = tool_path_table.linear_interpolation(t, Fields::x, Fields::time);
        // get the y position at time t
        y = tool_path_table.linear_interpolation(t, Fields::y, Fields::time);
        // get the z position at time t
        z = tool_path_table.linear_interpolation(t, Fields::z, Fields::time);
    } // end function

    // Luther - added a function to get only the z coordinate of the heat source on the host
    // Compute current position of tool at time t, assuming linear motion between path points.
    // Returns a double, z, and accesses on host side

    // Should this not be KOKKOS_INLINE_FUNCTION since it's on the host side?
    KOKKOS_INLINE_FUNCTION
    void get_position_z_host(const double& t, double& z) const {

        // get the z position at time t
        z = tool_path_table.linear_interpolation_host(t, Fields::z, Fields::time);
        
    } // end function

    // Luther - added a funcion to get only the z coordinate of the heat source on the device
    // Compute current position of tool at time t, assuming linear motion between path points.
    // Returns a double, z, and accesses on device side
    KOKKOS_INLINE_FUNCTION
    void get_position_z_device(const double& t, double& z) const {

        // get the z position at time t
        z = tool_path_table.linear_interpolation(t, Fields::z, Fields::time);
        
    } // end function


    // Compute current power of tool at time t on the device, assuming linear interpolation between path points.
    // Returns the power at the time t
    KOKKOS_INLINE_FUNCTION
    double get_power(const double& t) const {
        return tool_path_table.linear_interpolation(t, Fields::power, Fields::time);
    } // end function
};

// ----------------------------------
// valid inputs for additive data
// ----------------------------------
static std::vector<std::string> str_additive_data_inps
{
    "time_initial",
    "time_final",
    "dt_min",
    "dt_max",
    "dt_start",
    "dt_cfl",
    "cycle_stop",
    "fuzz",
    "tiny",
    "small",
    "rk_num_stages"
};

// ----------------------------------
// required inputs for dynamic options
// ----------------------------------
static std::vector<std::string> additive_data_required_inps
{
    "time_final",
    "cycle_stop",
};

#endif // end Header Guard