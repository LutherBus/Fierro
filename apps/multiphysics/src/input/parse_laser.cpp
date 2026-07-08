// Luther - added a parser to read the laser path and laser shape parameters from the input file

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
    "path",
    "model",
    "radius",
    "absorptivity",
    "major_front",
    "major_rear",
    "minor",
    "depth",
};
std::vector<std::string> laser_path_required_inps = {
    "path",
    "model"
};

void parse_laser(Yaml::Node& root, Laser_t& Laser)
{
    Yaml::Node& laser_path_yaml = root["laser_options"];
    // get the laser_path variable names set by the user
    std::vector<std::string> user_str_laser_path_inps;
    // extract words from the input file and validate they are correct
    validate_inputs(laser_path_yaml, user_str_laser_path_inps, str_laser_path_inps, laser_path_required_inps);

    // loop over the words in the laser_path input definition
    std::string laser_path_file = "";
    bool radius_set = false;
    bool absorptivity_set = false;
    bool major_front_set = false;
    bool major_rear_set = false;
    bool minor_set = false;
    bool depth_set = false;

    for (auto& a_word : user_str_laser_path_inps) {
        Yaml::Node& laser_path_inps_yaml = root["laser_options"][a_word];

        if (a_word.compare("path") == 0) {
            laser_path_file = root["laser_options"]["path"].As<std::string>();

            // validate BEFORE storing into Laser
            if (laser_path_file.empty()) {
                std::cout << "ERROR: invalid laser path specified in the laser_options definition " << std::endl;
                throw std::runtime_error("**** laser path is empty ****");
            } // end check on empty path

            // optional: verify the file actually exists on disk
            std::ifstream file_check(laser_path_file);
            if (!file_check.good()) {
                std::cout << "ERROR: laser_options path file not found: " << laser_path_file << std::endl;
                throw std::runtime_error("**** laser_options path file does not exist ****");
            } // end file existence check

            // only assign once validated
            std::cout << "Assigning laser path in parse_laser.cpp" << std::endl;
            Laser.path_file = laser_path_file;
            Laser.tool_path_info = ToolPathInfo::load_laser_path(laser_path_file);
        } // end laser_path
        else if (a_word.compare("model") == 0) {
            std::string model_str = root["laser_options"]["model"].As<std::string>();

            if (model_str.compare("spherical") == 0) {
                Laser.model = LaserModelType::SPHERICAL;
            }
            else if (model_str.compare("goldak") == 0) {
                Laser.model = LaserModelType::GOLDAK;
            }
            else {
                std::cout << "ERROR: invalid laser model specified in the laser_options definition: "
                           << model_str << std::endl;
                throw std::runtime_error("**** laser_options model must be 'spherical' or 'goldak' ****");
            } // end model validation

            std::cout << "Assigning laser model in parse_laser.cpp: " << model_str << std::endl;
        } // end model
        else if (a_word.compare("radius") == 0) {
            double radius = root["laser_options"]["radius"].As<double>();

            if (radius <= 0.0) {
                std::cout << "ERROR: invalid radius specified in the laser_options definition " << std::endl;
                throw std::runtime_error("**** laser_options radius must be positive ****");
            } // end check on radius

            std::cout << "Assigning laser radius in parse_laser.cpp" << std::endl;
            Laser.spherical.radius = radius;
            radius_set = true;
        } // end radius
        else if (a_word.compare("absorptivity") == 0) {
            double absorptivity = root["laser_options"]["absorptivity"].As<double>();

            if (absorptivity <= 0.0 || absorptivity > 1.0) {
                std::cout << "ERROR: invalid absorptivity  specified in the laser_options definition " << std::endl;
                throw std::runtime_error("**** laser_options absorptivity must be in (0, 1] ****");
            } // end check on n

            Laser.goldak.absorptivity = absorptivity;
            absorptivity_set = true;
        } // end n
        else if (a_word.compare("major_front") == 0) {
            double major_front = root["laser_options"]["major_front"].As<double>();

            if (major_front <= 0.0) {
                std::cout << "ERROR: invalid major_front axis specified in the laser_options definition " << std::endl;
                throw std::runtime_error("**** laser_options major_front must be positive ****");
            } // end check on a_f

            Laser.goldak.major_front = major_front;
            major_front_set = true;
        } // end major_front
        else if (a_word.compare("major_rear") == 0) {
            double major_rear= root["laser_options"]["major_rear"].As<double>();

            if (major_rear <= 0.0) {
                std::cout << "ERROR: invalid major_rear specified in the laser_options definition " << std::endl;
                throw std::runtime_error("**** laser_options major_rear must be positive ****");
            } // end check on major_rear

            Laser.goldak.major_rear = major_rear;
            major_rear_set = true;
        } // end major_rear
        else if (a_word.compare("minor") == 0) {
            double minor = root["laser_options"]["minor"].As<double>();

            if (minor <= 0.0) {
                std::cout << "ERROR: invalid minor axis specified in the laser_options definition " << std::endl;
                throw std::runtime_error("**** laser_options minor axis must be positive ****");
            } // end check on minor

            Laser.goldak.minor = minor;
            minor_set = true;
        } // end minor
        else if (a_word.compare("depth") == 0) {
            double depth = root["laser_options"]["depth"].As<double>();

            if (depth <= 0.0) {
                std::cout << "ERROR: invalid depth specified in the laser_options definition " << std::endl;
                throw std::runtime_error("**** laser_options depth must be positive ****");
            } // end check on depth

            Laser.goldak.depth = depth;
            depth_set = true;
        } // end depth
    } // end loop over all laser_path inputs

    // radius is required for the spherical model
    if (Laser.model == LaserModelType::SPHERICAL && !radius_set) {
        std::cout << "ERROR: laser model is 'spherical' but no radius was specified " << std::endl;
        throw std::runtime_error("**** laser_options radius is required for spherical model ****");
    }

    // absorptivity, major_front, major_rear, minor, and depth are all required for the goldak model
    if (Laser.model == LaserModelType::GOLDAK) {
        if (!absorptivity_set) {
            std::cout << "ERROR: laser model is 'goldak' but no absorptivity was specified " << std::endl;
            throw std::runtime_error("**** laser_options absorptivity is required for goldak model ****");
        }
        if (!major_front_set) {
            std::cout << "ERROR: laser model is 'goldak' but no major_front axis was specified " << std::endl;
            throw std::runtime_error("**** laser_options major_front axis is required for goldak model ****");
        }
        if (!major_rear_set) {
            std::cout << "ERROR: laser model is 'goldak' but no major_rear axis was specified " << std::endl;
            throw std::runtime_error("**** laser_options major_rear axis is required for goldak model ****");
        }
        if (!minor_set) {
            std::cout << "ERROR: laser model is 'goldak' but no minor axis was specified " << std::endl;
            throw std::runtime_error("**** laser_options minor axis is required for goldak model ****");
        }
        if (!depth_set) {
            std::cout << "ERROR: laser model is 'goldak' but no depth was specified " << std::endl;
            throw std::runtime_error("**** laser_options depth is required for goldak model ****");
        }
    } // end goldak required-param checks
} // end parse_laser_path