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

#include "sgtm_solver_3D.hpp"
#include "material.hpp"
//#include "mesh.hpp""
#include "state.hpp"
#include "geometry_new.hpp"
#include "additive_data.hpp"


/////////////////////////////////////////////////////////////////////////////
///
/// \fn get_heat_flux
///
/// \brief This function calculates the corner heat flux
///
/// \param Materials in the simulation
/// \param The simulation mesh
/// \param Gauss point (element) volume
/// \param Nodal position array
/// \param Nodal temperature array
/// \param Material point heat flux array
/// \param Material point thermal conductivity
/// \param Material Point temperature gradient
/// \param Material Point state variables
/// \param Corner heat flux
/// \param Material corner heat flux array
/// \param Map from material to corners
/// \param Maps from the material to the mesh
/// \param Number of elements associated with a given material
/// \param Material ID
/// \param fuzz
/// \param small
/// \param The timestep
/// \param The current Runge Kutta integration alpha value
///
/////////////////////////////////////////////////////////////////////////////
void SGTM3D::get_heat_flux(
    const Material_t& Materials,
    const swage::Mesh& mesh,
    const DCArrayKokkos<double>& GaussPoints_vol,
    const MPICArrayKokkos<double>& node_coords,
    const MPICArrayKokkos<double>& node_temp,
    const DRaggedRightArrayKokkos<double>& MaterialPoints_q_flux,
    const DRaggedRightArrayKokkos<double>& MaterialPoints_conductivity,
    const DRaggedRightArrayKokkos<double>& MaterialPoints_temp_grad,
    const DCArrayKokkos<double>& corner_q_transfer,
    const corners_in_mat_t corners_in_mat_elem,
    const DRaggedRightArrayKokkos<bool>&   MaterialPoints_eroded,
    const DRaggedRightArrayKokkos<size_t>& elem_in_mat_elem,
    const size_t num_mat_elems,
    const size_t mat_id,
    const double fuzz,
    const double small,
    const double dt,
    const double rk_alpha,
    DynamicArrayKokkos<size_t>& mat_elem_sid_activated) const
{
    const size_t num_dims = 3;
    const size_t num_nodes_in_elem = 8;

    // ---- calculate the forces acting on the nodes from the element ---- //
    FOR_ALL(i, 0, mat_elem_sid_activated.dims(0), {

        // get elem gid
        size_t elem_gid = elem_in_mat_elem(mat_id, mat_elem_sid_activated(i)); 

        // the material point index = the material elem index for a 1-point element
        size_t mat_point_sid = mat_elem_sid_activated(i);

        // corner area normals
        double b_matrix_array[24];
        ViewCArrayKokkos<double> b_matrix(b_matrix_array, num_nodes_in_elem, num_dims);

        // temperature gradient
        double temp_grad_array[3];
        ViewCArrayKokkos<double> temp_grad(&temp_grad_array[0], 3);

        // element volume
        double vol = GaussPoints_vol(elem_gid);
        // printf("Elem vol = %e\n", vol);

        // cut out the node_gids for this element
        ViewCArrayKokkos<size_t> elem_node_gids(&mesh.nodes_in_elem(elem_gid, 0), 8);

        // ---- get the B matrix which are the OUTWARD corner area normals ---- //
        geometry::get_bmatrix(b_matrix, elem_gid, node_coords, elem_node_gids);


        // ---- Calculate the element average temperature ---- //
        double avg_temp = 0.0;
        for (size_t node_lid = 0; node_lid < num_nodes_in_elem; node_lid++) {
            // Get node gid
            size_t node_gid = elem_node_gids(node_lid);
            avg_temp += node_temp(node_gid) / (double)num_nodes_in_elem;
        } // end for


        // ---- Change element state if above some melting temperature ---- //
        if(avg_temp >= 1600){
            // printf("Melted!");
            MaterialPoints_eroded(mat_id, mat_elem_sid_activated(i)) = true;
        } 

        // ---- Calculate the temperature gradient ---- //
        double inverse_vol = 1.0 / vol;

        temp_grad(0) = 0.0;
        temp_grad(1) = 0.0;
        temp_grad(2) = 0.0;

        for(int dim = 0; dim < mesh.num_dims; dim++){
            for (size_t node_lid = 0; node_lid < num_nodes_in_elem; node_lid++) {
                // Get node gid
                size_t node_gid = elem_node_gids(node_lid);

                temp_grad(dim) += node_temp(node_gid) * b_matrix(node_lid, dim); // Note: B matrix is outward normals from cell center
            }
        }
        for(int dim = 0; dim < mesh.num_dims; dim++){
            temp_grad(dim) *= inverse_vol;
        }

        // ---- Save the temperature gradient to the material point for writing out ---- //
        MaterialPoints_temp_grad(mat_id, elem_gid, 0) = temp_grad(0);
        MaterialPoints_temp_grad(mat_id, elem_gid, 1) = temp_grad(1);
        MaterialPoints_temp_grad(mat_id, elem_gid, 2) = temp_grad(2);

        // ---- Calculate the heat flux at the material point ---- //
        double conductivity = MaterialPoints_conductivity(mat_id, mat_point_sid); // NOTE: Consider moving this to properties and evaluate instead of save
        for(int dim = 0; dim < mesh.num_dims; dim++){
            MaterialPoints_q_flux(mat_id, mat_point_sid, dim) = -1.0 * conductivity * temp_grad(dim);
        }

        // --- Calculate flux through each corner the corners \lambda_{c} = q_z \cdot \hat B_c   ---- //
        for (size_t node_lid = 0; node_lid < num_nodes_in_elem; node_lid++) {
            // the local corner id is the local node id
            size_t corner_lid = node_lid;

            // Get corner gid
            size_t corner_gid = mesh.corners_in_elem(elem_gid, corner_lid);

            // Get the material corner lid
            size_t mat_corner_lid = corners_in_mat_elem(mat_elem_sid_activated(i), corner_lid);

            // Zero out flux at material corners
            corner_q_transfer(corner_gid) = 0.0;

            // Dot the flux into the corner normal
            for(int dim = 0; dim < mesh.num_dims; dim++){
                corner_q_transfer(corner_gid) += MaterialPoints_q_flux(mat_id, mat_point_sid, dim) * (1.0*b_matrix(node_lid, dim));
            }
        }
    }); // end parallel for loop over elements associated with the given material

    return;
} // end of routine






/////////////////////////////////////////////////////////////////////////////
///
/// \fn moving flux
///
/// \brief 
///
/// \param Materials in the simulation
/// \param The simulation mesh
/// \param Gauss point (element) volume
/// \param Nodal position array
/// \param Nodal temperature array
/// \param Material point heat flux array
/// \param Material Point state variables
/// \param Material corner heat flux array
/// \param Map from material to corners
/// \param Maps from the material to the mesh
/// \param Number of elements associated with a given material
/// \param Material ID
/// \param fuzz
/// \param small
/// \param Element state variable array
/// \param Time step size
/// \param The current Runge Kutta integration alpha value
///
/////////////////////////////////////////////////////////////////////////////
void SGTM3D::moving_flux(
    const Material_t& Materials,
    const swage::Mesh& mesh,
    const DCArrayKokkos<double>& GaussPoints_vol,
    const MPICArrayKokkos<double>& node_coords,
    const DCArrayKokkos<double>& corner_q_flux,
    const corners_in_mat_t corners_in_mat_elem,
    const DRaggedRightArrayKokkos<size_t>& elem_in_mat_elem,
    const size_t num_mat_elems,
    const size_t mat_id,
    const double fuzz,
    const double small,
    const double dt,
    const double rk_alpha,
    const double time_value,
    const ToolPathInfo& path,
    DynamicArrayKokkos<size_t>& mat_elem_sid_activated) const
{

    // ---- Apply heat flux from a moving heat source ---- //
    FOR_ALL(i, 0, mat_elem_sid_activated.dims(0), {
        
        // get elem gid
        size_t elem_gid = elem_in_mat_elem(mat_id, mat_elem_sid_activated(i)); 
    
        // calculate the coordinates and radius of the element
        double elem_coords_1D[3]; // note:initialization with a list won't work
        ViewCArrayKokkos<double> elem_coords(&elem_coords_1D[0], 3);
        elem_coords(0) = 0.0;
        elem_coords(1) = 0.0;
        elem_coords(2) = 0.0;

        // get the coordinates of the element center (using rk_level=1 or node coords)
        for (int node_lid = 0; node_lid < mesh.num_nodes_in_elem; node_lid++) {
            elem_coords(0) += node_coords(mesh.nodes_in_elem(elem_gid, node_lid), 0);
            elem_coords(1) += node_coords(mesh.nodes_in_elem(elem_gid, node_lid), 1);
            elem_coords(2) += node_coords(mesh.nodes_in_elem(elem_gid, node_lid), 2);
        } // end loop over nodes in element
        elem_coords(0) = (elem_coords(0) / mesh.num_nodes_in_elem);
        elem_coords(1) = (elem_coords(1) / mesh.num_nodes_in_elem);
        elem_coords(2) = (elem_coords(2) / mesh.num_nodes_in_elem);
        
        // Gaussian semi-elipsoid heat source model from (Goldak, 1984)

        // Q(x,y,z) = 10.39230 * f_f * n * power / (a * b * c_f * 5.568328) 
        //              * exp[-3 * (d_x * d_x) / (a_f * a_f) + (d_y * d_y) / (b * b) + (dz * dz) / (c * c)] for d_x >= 0
        
        // Q(x,y,z) = 10.39230 * f_r * n * power / (a * b * c_r * 5.568328)
        //               * exp[-3 * (d_x * d_x) / (a * a) + (d_y * d_y) / (b * b) + (dz * dz) / (c * c)] for d_x <= 0

        double power = path.get_power(time_value);
        double n = 0.67; // Absorptivity of powder bed
        double a_f = 0.02; // Semi-axis along travel direction, front (mm)
        double a_r = 0.06; // Semi-axis along travel direction, rear (mm)
        double b = 0.04; // Transverse semi_axis (mm)
        double c = 0.08; // Depth (mm) 
        double f_f = 2.0 * a_f / (a_f + a_r);  // Heat fraction, front
        double f_r = 2.0 * a_r / (a_f + a_r);  // Heat fraction, rear

        // Calculate the velocity of the heat source
        
        // Get future heat source position
        double x1 = 0.0;
        double y1 = 0.0;
        double z1 = 0.0;
        path.get_position(time_value + dt, x1, y1, z1);


        // Get previous heat source position 
        double x_hs = 0.0;
        double y_hs = 0.0;
        double z_hs = 0.0;
        
        // Get current heat source position
        double x0 = 0.0;
        double y0 = 0.0;
        double z0 = 0.0;

        path.get_position(time_value, x_hs, y_hs, z_hs);

        // Check if it is the first timestep
        if (time_value - dt < 0) {
            x0 = x_hs;
            y0 = y_hs;
            z0 = z_hs;
        } else {
            path.get_position(time_value - dt, x0, y0, z0);
        } // end if/else for previous heat source position

        double L = Kokkos::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        double xi;
        double eta;
        double dz = elem_coords(2) - z_hs;

        // Check if heat source has moved
        if (L < 1e-12) {
            // No movement detected, default to global x/y
            xi  = elem_coords(0) - x_hs;
            eta = elem_coords(1) - y_hs;
        } else {
            // Rotate heat source using central differencing scheme 
            xi  =  (elem_coords(0) - x_hs) * (x1 - x0) / L
                + (elem_coords(1) - y_hs) * (y1 - y0) / L;
            eta = -(elem_coords(0) - x_hs) * (y1 - y0) / L
                + (elem_coords(1) - y_hs) * (x1 - x0) / L;
        }

        // Ellipsoid cutoff to reduce the number of elements that are impacted by the heat source
        double a_max = (xi >= 0.0) ? a_f : a_r;
        double ellipsoid_dist = (xi * xi) / (a_max * a_max)
                      + (eta * eta) / (b * b)
                      + (dz * dz)  / (c * c);

        // Compute volumetric heat flux for elements within the heat source
        if (ellipsoid_dist <= 9.0) {
            for (size_t node_lid = 0; node_lid < mesh.num_nodes_in_elem; node_lid++) {
                size_t node_gid = mesh.nodes_in_elem(elem_gid, node_lid);

                // the local corner id is the local node id
                size_t corner_lid = node_lid;

                // Get corner gid
                size_t corner_gid = mesh.corners_in_elem(elem_gid, corner_lid);

                // Calculate the volumetric heat flux depending on the direction in which the heat source is moving
                double q_dot = 0.0;
                if (xi >= 0) {
                    q_dot = 10.39230 * f_f * n * power / (a_f * b * c * 5.568328) 
                            * Kokkos::exp(-3 * ((xi * xi) / (a_f * a_f) + (eta * eta) / (b * b) + (dz * dz) / (c * c)));
                // std::cout << "q_dot = " << q_dot << std::endl;
                } else {
                    q_dot = 10.39230 * f_r * n * power / (a_r * b * c * 5.568328)
                            * Kokkos::exp(-3 * ((xi * xi) / (a_r * a_r) + (eta * eta) / (b * b) + (dz * dz) / (c * c)));
                } // end if/else for computing the volumetric heat flux

                // Note: this will be 1/8th the volumetric flux times the volume
                corner_q_flux(corner_gid) += q_dot * 0.125 * GaussPoints_vol(elem_gid);

                // std::cout << "corner_q_flux = " << corner_q_flux(corner_gid) << std::endl;
                //std::cout << "flux delta = " << q_dot * 0.125 * GaussPoints_vol(elem_gid) * 1000000.0 << std::endl;

            } // end for loop
            
        } // end loop over elements within the heat source boundary

    }); // end parallel for loop over elements

    // Note: a correction term may be needed to account for the fact that the flux is not evenly distributed to enforce conservation

    return;
}

