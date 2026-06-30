#include <string>      // std::string
#include <vector>      // std::vector
#include <iostream>    // std::cout
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

#include "laser.hpp"

#include "string_utils.hpp"
#include "matar.h"
#include "parse_tools.hpp"
#include "Yaml.hpp"
#include "additive_data.hpp"

Laser_t Laser;

std::vector<std::string> str_laser_path_inps = {
    "tabular_data"
};

std::vector<std::string> laser_path_required_inps = {
    "tabular_data"
};

void parse_laser_path(Yaml::Node& root, Laser_t& Laser)
{


    // read the laser_path block (single block, not a list like materials)
    Yaml::Node& laser_path_yaml = root["laser_path"];

    // get the laser_path variable names set by the user
    std::vector<std::string> user_str_laser_path_inps;

    // extract words from the input file and validate they are correct
    validate_inputs(laser_path_yaml, user_str_laser_path_inps, str_laser_path_inps, laser_path_required_inps);

    // loop over the words in the laser_path input definition
    std::string laser_path_file = "";
    for (auto& a_word : user_str_laser_path_inps) {
        Yaml::Node& laser_path_inps_yaml = root["laser_path"][a_word];

        if (a_word.compare("tabular_data") == 0) {
            laser_path_file = root["laser_path"]["tabular_data"].As<std::string>();

            // validate BEFORE storing into Laser
            if (laser_path_file.empty()) {
                std::cout << "ERROR: invalid tabular_data path specified in the laser_path definition " << std::endl;
                throw std::runtime_error("**** laser_path tabular_data is empty ****");
            } // end check on empty path

            // optional: verify the file actually exists on disk
            std::ifstream file_check(laser_path_file);
            if (!file_check.good()) {
                std::cout << "ERROR: laser_path tabular_data file not found: " << laser_path_file << std::endl;
                throw std::runtime_error("**** laser_path tabular_data file does not exist ****");
            } // end file existence check

            // only assign once validated
            std::cout << "Assigning laser path in parse_laser_path.cpp" << std::endl;
            Laser.path_file = laser_path_file;
            Laser.tool_path_info = ToolPathInfo::load_laser_path(laser_path_file);
        } // end tabular_data
    } // end loop over all laser_path inputs
} // end parse_laser_path