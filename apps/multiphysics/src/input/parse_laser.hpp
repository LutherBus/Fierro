#ifndef PARSE_LASER_HPP
#define PARSE_LASER_HPP

#include <vector>
#include <string>

#include "Yaml.hpp"
#include "laser.hpp"

/**
 * @brief Parses the "laser_path" block from the input YAML file.
 *
 * Reads the laser_path inputs (currently the "tabular_data" key), validates
 * that the specified tabular data file path is non-empty and exists on disk,
 * then stores the validated path and loads the parsed tool path data into
 * the global Laser object.
 *
 * @param root Root YAML node of the input file.
 *
 * @throws std::runtime_error if tabular_data is missing/empty or the file
 *         does not exist.
 */

extern std::vector<std::string> str_laser_path_inps;
extern std::vector<std::string> laser_path_required_inps;


void parse_laser(Yaml::Node& root, Laser_t& Laser);

#endif