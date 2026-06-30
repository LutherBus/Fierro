/**********************************************************************************************
© 2020. Triad National Security, LLC. All rights reserved.
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

#include "sgtm_solver_3D.hpp"

#include "simulation_parameters.hpp"
#include "material.hpp"
#include "boundary_conditions.hpp"
#include "state.hpp"
#include "geometry_new.hpp"
#include "mesh_io.hpp"
#include "additive_data.hpp"

using std::cout;
using std::endl;

/////////////////////////////////////////////////////////////////////////////
///
/// \fn execute
///
/// Evolve the state according to the SGH method
///
/////////////////////////////////////////////////////////////////////////////
void SGTM3D::execute(SimulationParameters_t& SimulationParamaters, 
                     Material_t& Materials, 
                     BoundaryCondition_t& BoundaryConditions, 
                     swage::Mesh& mesh, 
                     State_t& State) {

    double fuzz  = SimulationParamaters.DynamicOptions.fuzz;
    double tiny  = SimulationParamaters.DynamicOptions.tiny;
    double small = SimulationParamaters.DynamicOptions.small;

    double graphics_dt_ival  = SimulationParamaters.OutputOptions.graphics_time_step;
    int    graphics_cyc_ival = SimulationParamaters.OutputOptions.graphics_iteration_step;

    // double time_initial = SimulationParamaters.DynamicOptions.time_initial;
    double time_final = this->time_end; // SimulationParamaters.DynamicOptions.time_final;
    double dt_min   = SimulationParamaters.DynamicOptions.dt_min;
    double dt_max   = SimulationParamaters.DynamicOptions.dt_max;
    double dt_start = SimulationParamaters.DynamicOptions.dt_start;
    double dt_cfl   = SimulationParamaters.DynamicOptions.dt_cfl;

    int rk_num_stages = SimulationParamaters.DynamicOptions.rk_num_stages;
    int cycle_stop    = SimulationParamaters.DynamicOptions.cycle_stop;

    // initialize time, time_step, and cycles
    double time_value = this->time_start; // 0.0;
    double dt = dt_start;

    // Create mesh writer
    MeshWriter mesh_writer; // Note: Pull to driver after refactoring evolution

    // --- Graphics vars ----
    CArray<double> graphics_times = CArray<double>(20000);
    graphics_times(0) = this->time_start; // was zero
    double graphics_time = this->time_start; // the times for writing graphics dump, was zero
    size_t output_id=0; // the id for the outputs written
    
    boundary_temperature(mesh, BoundaryConditions, State.node.temp, time_value); // Time value = 0.0;

    double cached_pregraphics_dt = fuzz;

    // the number of materials specified by the user input
    const size_t num_mats = Materials.num_mats;

    // a flag to exit the calculation
    size_t stop_calc = 0;
    auto time_1 = std::chrono::high_resolution_clock::now();    

    // ---- Initialize the tool path information ---- //
    int number_of_points = 18;
    ToolPathInfo path(number_of_points);

    // This toolpath is set to 700 mm/s, a typical scan speed
    // The laser power is set to 150 watts
    /* path.set_data_point(0, 0.00000000, 0.300000, 0.250000, 0.025, 150000.0);
    path.set_data_point(1, 0.00002061, 0.295500, 0.260900, 0.025, 150000.0);
    path.set_data_point(2, 0.00004122, 0.294700, 0.272300, 0.025, 150000.0);
    path.set_data_point(3, 0.00006184, 0.297600, 0.284000, 0.025, 150000.0);
    path.set_data_point(4, 0.00008245, 0.304200, 0.295300, 0.025, 150000.0);
    path.set_data_point(5, 0.00010306, 0.314300, 0.305700, 0.025, 150000.0);
    path.set_data_point(6, 0.00012367, 0.327000, 0.315000, 0.025, 150000.0);
    path.set_data_point(7, 0.00014429, 0.341600, 0.323400, 0.025, 150000.0);
    path.set_data_point(8, 0.00016490, 0.357600, 0.331000, 0.025, 150000.0);
    path.set_data_point(9, 0.00018551, 0.375100, 0.338000, 0.025, 150000.0);
    path.set_data_point(10, 0.00020612, 0.394100, 0.344200, 0.025, 150000.0);
    path.set_data_point(11, 0.00022673, 0.414300, 0.348700, 0.025, 150000.0);
    path.set_data_point(12, 0.00024735, 0.435300, 0.348100, 0.025, 150000.0);
    path.set_data_point(13, 0.00026796, 0.456400, 0.340800, 0.025, 150000.0);
    path.set_data_point(14, 0.00028857, 0.477100, 0.326500, 0.025, 150000.0);
    path.set_data_point(15, 0.00030918, 0.497100, 0.307400, 0.025, 150000.0);
    path.set_data_point(16, 0.00032980, 0.516200, 0.287700, 0.025, 150000.0);
    path.set_data_point(17, 0.00035041, 0.535000, 0.270300, 0.025, 150000.0);
    path.set_data_point(18, 0.00037102, 0.554100, 0.256700, 0.025, 150000.0);
    path.set_data_point(19, 0.00039163, 0.573800, 0.246300, 0.025, 150000.0);
    path.set_data_point(20, 0.00041224, 0.594300, 0.237900, 0.025, 150000.0);
    path.set_data_point(21, 0.00043286, 0.615400, 0.230000, 0.025, 150000.0);
    path.set_data_point(22, 0.00045347, 0.636900, 0.221900, 0.025, 150000.0);
    path.set_data_point(23, 0.00047408, 0.658500, 0.213100, 0.025, 150000.0);
    path.set_data_point(24, 0.00049469, 0.679800, 0.203300, 0.025, 150000.0);
    path.set_data_point(25, 0.00051531, 0.700900, 0.192800, 0.05, 150000.0);
    path.set_data_point(26, 0.00053592, 0.721800, 0.181900, 0.05, 150000.0);
    path.set_data_point(27, 0.00055653, 0.742600, 0.170900, 0.05, 150000.0);
    path.set_data_point(28, 0.00057714, 0.763400, 0.160400, 0.05, 150000.0);
    path.set_data_point(29, 0.00059776, 0.784400, 0.151200, 0.05, 150000.0);
    path.set_data_point(30, 0.00061837, 0.805700, 0.145000, 0.05, 150000.0);
    path.set_data_point(31, 0.00063898, 0.827400, 0.143000, 0.05, 150000.0);
    path.set_data_point(32, 0.00065959, 0.849300, 0.145800, 0.05, 150000.0);
    path.set_data_point(33, 0.00068020, 0.871400, 0.153400, 0.05, 150000.0);
    path.set_data_point(34, 0.00070082, 0.893500, 0.165000, 0.05, 150000.0);
    path.set_data_point(35, 0.00072143, 0.915300, 0.179400, 0.05, 150000.0);
    path.set_data_point(36, 0.00074204, 0.936500, 0.195300, 0.05, 150000.0);
    path.set_data_point(37, 0.00076265, 0.956700, 0.211100, 0.05, 150000.0);
    path.set_data_point(38, 0.00078327, 0.975400, 0.225200, 0.05, 150000.0);
    path.set_data_point(39, 0.00080388, 0.992300, 0.236400, 0.05, 150000.0);
    path.set_data_point(40, 0.00082449, 1.000000, 0.250000, 0.05, 150000.0);
    path.set_data_point(41, 0.00084510, 1.000000, 0.250000, 0.05, 150000.0);
    path.set_data_point(42, 0.00086571, 1.000000, 0.250000, 0.05, 150000.0);
    path.set_data_point(43, 0.00088633, 1.000000, 0.250000, 0.05, 150000.0);
    path.set_data_point(44, 0.00090694, 1.000000, 0.250000, 0.05, 150000.0);
    path.set_data_point(45, 0.00092755, 1.000000, 0.250000, 0.05, 150000.0);
    path.set_data_point(46, 0.00094816, 1.000000, 0.250000, 0.05, 150000.0);
    path.set_data_point(47, 0.00096878, 1.000000, 0.250000, 0.05, 150000.0);
    path.set_data_point(48, 0.00098939, 1.000000, 0.250000, 0.05, 150000.0);
    path.set_data_point(49, 0.00101000, 1.000000, 0.250000, 0.05, 0.0);
    */
    /*
    path.set_data_point(0, 0.000, 0.3, 0.25, 0.05, 150000.0);
    path.set_data_point(1, 0.001, 1.0, 0.25, 0.05, 150000.0);
    path.set_data_point(2, 0.001, 1.0, 0.25, 0.05, 0.0);
    
    path.set_data_point(3, 0.002, 0.3, 0.25, 0.1, 0.0);
    path.set_data_point(4, 0.003, 0.3, 0.25, 0.1, 0.0);
    path.set_data_point(5, 0.003, 0.3, 0.25, 0.1, 150000.0);
    path.set_data_point(6, 0.004, 1.0, 0.25, 0.1, 150000.0);
    path.set_data_point(7, 0.004, 1.0, 0.25, 0.1, 0.0);
   
    path.set_data_point(8, 0.005, 0.3, 0.25, 0.15, 0.0);
    path.set_data_point(9, 0.006, 0.3, 0.25, 0.15, 0.0);
    path.set_data_point(10, 0.006, 0.3, 0.25, 0.15, 150000.0);
    path.set_data_point(11, 0.007, 1.0, 0.25, 0.15, 150000.0);
    path.set_data_point(12, 0.007, 1.0, 0.25, 0.15, 0.0);
    
    path.set_data_point(13, 0.008, 0.3, 0.25, 0.2, 0.0);
    path.set_data_point(14, 0.009, 0.3, 0.25, 0.2, 0.0);
    path.set_data_point(15, 0.009, 0.3, 0.25, 0.2, 150000.0);
    path.set_data_point(16, 0.01, 1.0, 0.25, 0.2, 150000.0);
    path.set_data_point(17, 0.01, 1.0, 0.25, 0.2, 0.0);
    
    path.set_data_point(18, 0.011, 0.3, 0.25, 0.25, 0.0);
    path.set_data_point(19, 0.012, 0.3, 0.25, 0.25, 0.0);
    path.set_data_point(20, 0.012, 0.3, 0.25, 0.25, 150000.0);
    path.set_data_point(21, 0.013, 1.0, 0.25, 0.25, 150000.0);
    path.set_data_point(22, 0.013, 1.0, 0.25, 0.25, 0.0);

    path.set_data_point(23, 0.014, 0.3, 0.25, 0.3, 0.0);
    path.set_data_point(24, 0.015, 0.3, 0.25, 0.3, 0.0);
    path.set_data_point(25, 0.015, 0.3, 0.25, 0.3, 150000.0);
    path.set_data_point(26, 0.016, 1.0, 0.25, 0.3, 150000.0);
    path.set_data_point(27, 0.016, 1.0, 0.25, 0.3, 0.0);

    path.set_data_point(28, 0.017, 0.3, 0.25, 0.35, 0.0);
    path.set_data_point(29, 0.018, 0.3, 0.25, 0.35, 0.0);
    path.set_data_point(30, 0.018, 0.3, 0.25, 0.35, 150000.0);
    path.set_data_point(31, 0.019, 1.0, 0.25, 0.35, 150000.0);
    path.set_data_point(32, 0.019, 1.0, 0.25, 0.35, 0.0);

    path.set_data_point(33, 0.020, 0.3, 0.25, 0.4, 0.0);
    path.set_data_point(34, 0.021, 0.3, 0.25, 0.4, 0.0);
    path.set_data_point(35, 0.021, 0.3, 0.25, 0.4, 150000.0);
    path.set_data_point(36, 0.022, 1.0, 0.25, 0.4, 150000.0);
    path.set_data_point(37, 0.022, 1.0, 0.25, 0.4, 0.0);
                        */

    path.set_data_point(0, 0.000, 0.3, 0.25, 0.025, 150000.0);
    path.set_data_point(1, 0.001, 1.0, 0.25, 0.025, 150000.0);
    path.set_data_point(2, 0.001, 1.0, 0.25, 0.025, 0.0);
    
    path.set_data_point(3, 0.002, 0.3, 0.25, 0.05, 0.0);
    path.set_data_point(4, 0.003, 0.3, 0.25, 0.05, 0.0);
    path.set_data_point(5, 0.003, 0.3, 0.25, 0.05, 150000.0);
    path.set_data_point(6, 0.004, 1.0, 0.25, 0.05, 150000.0);
    path.set_data_point(7, 0.004, 1.0, 0.25, 0.05, 0.0);
   
    path.set_data_point(8, 0.005, 0.3, 0.25, 0.075, 0.0);
    path.set_data_point(9, 0.006, 0.3, 0.25, 0.075, 0.0);
    path.set_data_point(10, 0.006, 0.3, 0.25, 0.075, 150000.0);
    path.set_data_point(11, 0.007, 1.0, 0.25, 0.075, 150000.0);
    path.set_data_point(12, 0.007, 1.0, 0.25, 0.075, 0.0);
    
    path.set_data_point(13, 0.008, 0.3, 0.25, 0.1, 0.0);
    path.set_data_point(14, 0.009, 0.3, 0.25, 0.1, 0.0);
    path.set_data_point(15, 0.009, 0.3, 0.25, 0.1, 150000.0);
    path.set_data_point(16, 0.01, 1.0, 0.25, 0.1, 150000.0);
    path.set_data_point(17, 0.01, 1.0, 0.25, 0.1, 0.0);
    path.tool_path_table.print_table();

    path.update_device();
    MATAR_FENCE();
        
    // ---- Initialize necessary variables for activating the elements/nodes below the heat source (in the z-direction) ---- //
    State.node.coords.update_host();
    const MPICArrayKokkos<double>& node_coords = State.node.coords;
    DRaggedRightArrayKokkos<size_t>& elem_in_mat_elem = State.MaterialToMeshMaps.elem_in_mat_elem;
    double z_coord = 0.0;
    path.get_position_z_host(time_value, z_coord);

    
    // ---- Initialize activated elements/nodes flag arrays ---- //
    DRaggedRightArrayKokkos<bool>& MaterialPoints_activated = State.MaterialPoints.activated;
    DCArrayKokkos<bool>& node_activated = State.node.activated;

    // ---- Initialize arrays to hold all activated elements/nodes ---- //
    DynamicArrayKokkos<size_t> mat_elem_sid_activated(State.MaterialToMeshMaps.num_mat_elems.host(0), "mat_elem_sid_activated");
    DynamicArrayKokkos<size_t> node_gid_activated(mesh.num_nodes, "node_gid_activated");

    // ---- Initialize the nodal activation to false ---- //
    FOR_ALL(node_gid, 0, mesh.num_nodes, {
        State.node.activated(node_gid) = false;
    }); // end for parallel for over nodes


    // ---- Initialize the element activation to false ---- //
    FOR_ALL(mat_elem_sid, 0, State.MaterialToMeshMaps.num_mat_elems.host(0), {
        State.MaterialPoints.activated(0, mat_elem_sid) = false;
    }); // end for parallel for over elements
    

    // ---- Calculate the z-coordinate for every element, and activate any elements below the current position of the heat source ---- //
    MATAR_FENCE();

    for(size_t mat_id = 0; mat_id < num_mats; mat_id++){
        MATAR_FENCE();
        int num_mat_elems = State.MaterialToMeshMaps.num_mat_elems.host(mat_id);

        for(size_t mat_elem_sid = 0; mat_elem_sid < num_mat_elems; mat_elem_sid++) {    
            MATAR_FENCE();
            size_t elem_gid = elem_in_mat_elem.host(mat_id, mat_elem_sid);
            
            ViewCArrayHost<size_t> elem_node_gids(&mesh.nodes_in_elem.host(elem_gid, mat_id), 8);
            
            // Get the z-coordinate of the element
            double avg_z = 0.0;

            for (size_t node_lid = 0; node_lid < 8; node_lid++) {
                avg_z += node_coords.host(mesh.nodes_in_elem.host(elem_gid, node_lid), 2);

            } // end for loop over node_lid

            avg_z *= 0.125;
            
            // Check if the element is below the z-coordinate of the heat source
            if (avg_z <= z_coord) {
                MaterialPoints_activated.host(mat_id, mat_elem_sid) = true; // If it is, set the activated flag for the element to true
                mat_elem_sid_activated.push_back(mat_elem_sid); // Add the element to the array of activated elements

                for (size_t node_lid = 0; node_lid < 8; node_lid++) { // Add the nodes of the newly activated element to the list of activated nodes if not already in it
                    if (!node_activated.host(elem_node_gids(node_lid))) { // Check if the nodes of the newly activated element are already activated
                        node_activated.host(elem_node_gids(node_lid)) = true; // If not, set the activated flag for the node to true
                        node_gid_activated.push_back(elem_node_gids(node_lid)); // Add the node to the array of activated nodes

                    } // end if loop for adding nodes to activated nodes array
                } // end for loop over all nodes in an activated element 
            } // end if statement to check if the element is below the heat source
        } // end for loop over mat_elem_sid
    } // end for loop over mat_id
    
    MATAR_FENCE();
    
    // ---- Update the device with the activated flags for elements and nodes ---- //
    MaterialPoints_activated.update_device();
    node_activated.update_device();
    


    // Print the material tables
    for(size_t mat_id = 0; mat_id < num_mats; mat_id++){
        if (log) log->info("Material %lu density table (solid):\n", mat_id);
        if (log) Materials.density_table_solid.print_table();
        if (log) log->info("Material %lu thermal conductivity table (solid):\n", mat_id);
        if (log) Materials.thermal_conductivity_table_solid.print_table();
        if (log) log->info("Material %lu specific heat table (solid):\n", mat_id);
        if (log) Materials.specific_heat_table_solid.print_table();
        if (log) log->info("Material %lu density table (liquid):\n", mat_id);
        if (log) Materials.density_table_liquid.print_table();
        if (log) log->info("Material %lu thermal conductivity table (liquid):\n", mat_id);
        if (log) Materials.thermal_conductivity_table_liquid.print_table();
        if (log) log->info("Material %lu specific heat table (liquid):\n", mat_id);
        if (log) Materials.specific_heat_table_liquid.print_table();
        if (log) log->info("Material %lu density table (powder):\n", mat_id);
        if (log) Materials.density_table_powder.print_table();
        if (log) log->info("Material %lu thermal conductivity table (powder):\n", mat_id);
        if (log) Materials.thermal_conductivity_table_powder.print_table();
        if (log) log->info("Material %lu specific heat table (powder):\n", mat_id);
        if (log) Materials.specific_heat_table_powder.print_table();
        if (log) log->flush();
    } // end for mat_id
    MATAR_FENCE();
    // ---- Write initial state at t=0 ---- 
    if (log) log->info("Writing outputs to file at %f \n", graphics_time);
    mesh_writer.write_mesh(
        mesh, 
        State,
        SimulationParamaters, 
        dt,
        time_value, 
        graphics_times,
        SGTM3D_State::required_node_state,
        SGTM3D_State::required_gauss_pt_state,
        SGTM3D_State::required_material_pt_state,
        this->solver_id);
    
    output_id++; // saved an output file

    graphics_time = time_value + graphics_dt_ival;


    // ---- Set up sphere to act as a moving heat source ---- //
    DCArrayKokkos<double> heat_source_position(3, "heat_source_position");

    heat_source_position.host(0) = 0.0;
    heat_source_position.host(1) = 0.0;
    heat_source_position.host(2) = 0.0;
    heat_source_position.update_device();

    MATAR_FENCE();


    FOR_ALL(node_gid, 0, mesh.num_nodes, {
        State.node.q_transfer(node_gid) = 0.0;
    }); // end for parallel for over nodes

    // ---- Update the material properties ---- //
    if (SimulationParamaters.solver_inputs[this->solver_id].use_moving_heat_source) {
        for(size_t mat_id = 0; mat_id < num_mats; mat_id++){
            update_properties(
                Materials, 
                mesh, 
                State.node.temp, 
                State.MaterialPoints.den, 
                State.MaterialPoints.conductivity, 
                State.MaterialPoints.specific_heat,
                State.MaterialPoints.eroded,
                State.MaterialPoints.activated, 
                State.MaterialToMeshMaps.elem_in_mat_elem, 
                State.MaterialToMeshMaps.num_mat_elems.host(mat_id), 
                mat_id,
                mat_elem_sid_activated);
        } // end for mat_id
    }

    MATAR_FENCE();

    // ---- loop over the max number of time integration cycles ---- //
    for (size_t cycle = 0; cycle < cycle_stop; cycle++) {

        // ---- stop calculation if flag ---- //
        if (stop_calc == 1) {
            break;
        }

        cached_pregraphics_dt = dt; // save the dt value for resetting after writing graphics dump

        double min_dt_calc = dt_max; // the smallest time step across all materials

        // ---- Calculating the maximum allowable time step ---- //
        for(size_t mat_id = 0; mat_id < num_mats; mat_id++){

            // initialize the material dt
            double dt_mat = dt;
            // ---- get the stable time step, both from CFL and Von Neumann stability ---- //
            get_timestep(mesh,
                         State.node.coords,
                         State.node.vel,
                         State.GaussPoints.vol,
                         State.MaterialPoints.sspd,
                         State.MaterialPoints.conductivity,
                         State.MaterialPoints.den,
                         State.MaterialPoints.specific_heat,
                         State.MaterialPoints.eroded,
                         State.MaterialToMeshMaps.elem_in_mat_elem,
                         State.MaterialToMeshMaps.num_mat_elems.host(mat_id),
                         time_value,
                         graphics_time,
                         time_final,
                         dt_max,
                         dt_min,
                         dt_cfl,
                         dt_mat,
                         fuzz,
                         tiny,
                         mat_id);

            // ---- save the smallest dt of all materials ---- //
            min_dt_calc = fmin(dt_mat, min_dt_calc);
        } // end for loop over all mats
        dt = min_dt_calc;

        // Global minimum dt across MPI ranks (each rank's CFL limit is local to its partition).
        {
            int init = 0;
            if (MPI_Initialized(&init) == MPI_SUCCESS && init) {
                MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            }
        }

        // Global minimum dt across MPI ranks (each rank's CFL limit is local to its partition).
        {
            int init = 0;
            if (MPI_Initialized(&init) == MPI_SUCCESS && init) {
                MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            }
        }

        // ---- Print the initial time step and time value ---- //
        if (cycle == 0) {
            if (log) log->info("cycle = %lu, time = %f, time step = %f \n", cycle, time_value, dt);
            if (log) log->info("cycle = %lu, time = %f, time step = %f \n", cycle, time_value, dt);
        }
        
        // ---- Print time step every 10 cycles ---- // 
        else if (cycle % 20 == 0) {
            if (log) log->info("cycle = %lu, time = %f, time step = %f \n", cycle, time_value, dt);
            if (log) log->info("cycle = %lu, time = %f, time step = %f \n", cycle, time_value, dt);
        } // end if



        // ---- Initialize the state for the RK integration scheme ---- //
            for(size_t mat_id = 0; mat_id < num_mats; mat_id++){
                // save the values at t = n
                rk_init(State.node.coords,
                    State.node.coords_n0,
                    State.node.vel,
                    State.node.vel_n0,
                    State.node.temp,
                    State.node.temp_n0,
                    State.node.q_transfer,
                    State.MaterialPoints.stress,
                    mesh.num_dims,
                    mesh.num_elems,
                    mesh.num_nodes,
                    State.MaterialPoints.num_material_points.host(mat_id));
            } // end for mat_id
            

        // ---- Integrate the solution forward to t(n+1) via Runge Kutta (RK) method ---- //
        for (size_t rk_stage = 0; rk_stage < rk_num_stages; rk_stage++) {

            double rk_alpha = 1.0 / ((double)rk_num_stages - (double)rk_stage);

            // ---- Initialize the nodal flux to zero for this RK stage ---- //
            FOR_ALL(node_gid, 0, mesh.num_nodes, {
                State.node.q_transfer(node_gid) = 0.0;
            }); // end for parallel for over nodes

            // ---- Calculate the corner heat flux from conduction per material ---- //
            for(size_t mat_id = 0; mat_id < num_mats; mat_id++){

                get_heat_flux(
                    Materials,
                    mesh,
                    State.GaussPoints.vol,
                    State.node.coords,
                    State.node.temp,  // fixed to use current time level
                    State.MaterialPoints.q_flux,
                    State.MaterialPoints.conductivity,
                    State.MaterialPoints.temp_grad,
                    State.corner.q_transfer,
                    State.corners_in_mat_elem,
                    State.MaterialPoints.eroded,
                    State.MaterialToMeshMaps.elem_in_mat_elem,
                    State.MaterialToMeshMaps.num_mat_elems.host(mat_id),
                    mat_id,
                    fuzz,
                    small,
                    dt, 
                    rk_alpha,
                    mat_elem_sid_activated);

                // ---- Calculate the corner heat flux from moving volumetric heat source ----
                if (SimulationParamaters.solver_inputs[this->solver_id].use_moving_heat_source) {
                    moving_flux(
                        Materials,
                        mesh,
                        State.GaussPoints.vol,
                        State.node.coords,
                        State.corner.q_transfer,
                        State.corners_in_mat_elem,
                        State.MaterialToMeshMaps.elem_in_mat_elem,
                        State.MaterialToMeshMaps.num_mat_elems.host(mat_id),
                        mat_id,
                        fuzz,
                        small,
                        dt, 
                        rk_alpha,
                        time_value,
                        path,
                        mat_elem_sid_activated);
                    
                    update_properties(
                        Materials, 
                        mesh, 
                        State.node.temp, 
                        State.MaterialPoints.den, 
                        State.MaterialPoints.conductivity, 
                        State.MaterialPoints.specific_heat,
                        State.MaterialPoints.eroded, 
                        State.MaterialPoints.activated,
                        State.MaterialToMeshMaps.elem_in_mat_elem, 
                        State.MaterialToMeshMaps.num_mat_elems.host(mat_id), 
                        mat_id,
                        mat_elem_sid_activated);
                }

            } // end for mat_id

            // ---- apply flux boundary conditions (convection/radiation)  ---- //
            boundary_convection(mesh, 
                                BoundaryConditions, 
                                State.node.temp, // fixed to use current time level
                                State.node.q_transfer, 
                                State.node.coords, // fixed to use current time level
                                time_value);
            boundary_radiation(mesh, 
                               BoundaryConditions, 
                               State.node.temp, 
                               State.node.q_transfer, 
                               State.node.coords, 
                               time_value);


            // ---- Update nodal temperature ---- //
            update_temperature(
                mesh,
                State.corner.q_transfer,
                State.node.temp,
                State.node.temp_n0,
                State.node.mass,
                State.node.q_transfer,
                State.MaterialPoints.specific_heat, // Note: Need to make this a node field, and calculate in the material loop
                rk_alpha,
                dt,
                node_gid_activated);


            // ---- apply temperature boundary conditions to the boundary patches----
            boundary_temperature(mesh, BoundaryConditions, State.node.temp, time_value);

            State.node.temp.communicate();
            State.node.temp_n0.communicate();

            State.node.temp.communicate();
            State.node.temp_n0.communicate();

            // ---- Find the element average temperature ---- //


            // ---- Calculate MaterialPoints state (stress) for next time step ---- //
            

            // ---- Calculate cell volume for next time step ---- //
            geometry::get_vol(State.GaussPoints.vol, State.node.coords, mesh);

            
        } // end of RK loop
        
        time_value += dt;

        // ---- Activate new elements, if needed ---- //
        
        // ---- Calculate the z-coordinate for every element, and activate any elements below the current position of the heat source ---- //
        MATAR_FENCE();

        double z_coord = 0.0;
        path.get_position_z_host(time_value, z_coord);

        for(size_t mat_id = 0; mat_id < num_mats; mat_id++){
            MATAR_FENCE();
            int num_mat_elems = State.MaterialToMeshMaps.num_mat_elems.host(mat_id);

            for(size_t mat_elem_sid = 0; mat_elem_sid < num_mat_elems; mat_elem_sid++) {    
                MATAR_FENCE();
                size_t elem_gid = elem_in_mat_elem.host(mat_id, mat_elem_sid);
                
                ViewCArrayHost<size_t> elem_node_gids(&mesh.nodes_in_elem.host(elem_gid, mat_id), 8);
                
                // Get the z-coordinate of the element
                double avg_z = 0.0;

                for (size_t node_lid = 0; node_lid < 8; node_lid++) {
                    avg_z += node_coords.host(mesh.nodes_in_elem.host(elem_gid, node_lid), 2);

                } // end for loop over node_lid

                avg_z *= 0.125;
                
                // Check if the element is below the z-coordinate of the heat source
                if (avg_z <= z_coord) {
                    if (!MaterialPoints_activated.host(mat_id, mat_elem_sid)) { // If it is, check if the element has already been activated
                        MaterialPoints_activated.host(mat_id, mat_elem_sid) = true; // If it has not previously been activated, set the activated flag for the element to true
                        mat_elem_sid_activated.push_back(mat_elem_sid); // Add the element to the array of activated elements

                        for (size_t node_lid = 0; node_lid < 8; node_lid++) { // Add the nodes of the newly activated element to the list of activated nodes if not already in it
                            if (!node_activated.host(elem_node_gids(node_lid))) { // Check if the nodes of the newly activated element are already activated
                                node_activated.host(elem_node_gids(node_lid)) = true; // If not, set the activated flag for the node to true
                                node_gid_activated.push_back(elem_node_gids(node_lid)); // Add the node to the array of activated nodes

                            } // end if statement for adding nodes to activated nodes array
                        } // end for loop over all nodes in an activated element
                    } // end if statement for checking if the element under the heat source has already been activated
                } // end if statement to check if the element is below the heat source
            } // end for loop over mat_elem_sid
        } // end for loop over mat_id
        
        MATAR_FENCE();
        
        // ---- Update the device with the activated flags for elements and nodes ---- //
        MaterialPoints_activated.update_device();
        node_activated.update_device();



        // ---- Move heat source ---- //
        if (SimulationParamaters.solver_inputs[this->solver_id].use_moving_heat_source) {
            RUN({
                double x = 0.0;
                double y = 0.0;
                double z = 0.0;
                path.get_position(time_value, x, y, z);
                heat_source_position(0) = x;
                heat_source_position(1) = y;
                heat_source_position(2) = z;
            });
        }
        // increment the time



        size_t write = 0;
        if ((cycle + 1) % graphics_cyc_ival == 0 && cycle > 0) {
            write = 1;
        }
        else if (cycle == cycle_stop) {
            write = 1;
        }
        else if (time_value >= time_final) {
            write = 1;
        }
        else if (time_value >= graphics_time) {
            write = 1;
        }

        // ---- Write outputs ---- //
        if (write == 1) {
            dt = cached_pregraphics_dt;
            if (log) log->info("Writing outputs to file at %f \n", graphics_time);
            if (log) log->info("cycle = %lu, time = %f, time step = %f \n", cycle, time_value, dt);
            if (log) log->flush();
            if (log) log->info("Writing outputs to file at %f \n", graphics_time);
            if (log) log->info("cycle = %lu, time = %f, time step = %f \n", cycle, time_value, dt);
            if (log) log->flush();
            mesh_writer.write_mesh(mesh,
                                   State,
                                   SimulationParamaters,
                                   dt,
                                   time_value,
                                   graphics_times,
                                   SGTM3D_State::required_node_state,
                                   SGTM3D_State::required_gauss_pt_state,
                                   SGTM3D_State::required_material_pt_state,
                                   this->solver_id);

            output_id++;
            graphics_time = (double)(output_id) * graphics_dt_ival;

        } // end if

        // end of calculation
        if (time_value >= time_final) {
            break;
        }
    } // end for cycle loop

    auto time_2    = std::chrono::high_resolution_clock::now();
    auto calc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time_2 - time_1).count();
    if (log) log->info("\nCalculation time in seconds: %f \n", calc_time * 1e-9);
    if (log) log->info("\nCalculation time in seconds: %f \n", calc_time * 1e-9);

} // end of SGH execute
