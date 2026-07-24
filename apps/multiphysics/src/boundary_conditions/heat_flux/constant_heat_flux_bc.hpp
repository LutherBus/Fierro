//Luther - added function to set nodal q_transfer values to heat flux boundary conditions on the boundary

#ifndef BOUNDARY_CONSTANT_HEAT_FLUX_H
#define BOUNDARY_CONSTANT_HEAT_FLUX_H

#include "boundary_conditions.hpp"

struct BoundaryConditionEnums_t;

namespace ConstantHeatFluxBC
{
/////////////////////////////////////////////////////////////////////////////
///
/// \fn heat flux
///
/// \brief This is a function to set the nodal heat fluxes along a symmetry 
///        plane or a wall. 
///
/// \param Mesh object
/// \param Boundary condition enums to select options
/// \param Boundary condition global variables array
/// \param Boundary condition state variables array
/// \param Node velocity
/// \param Time of the simulation
/// \param Boundary global index for the surface node
/// \param Boundary set local id
///
/////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
static void heat_flux(const swage::Mesh& mesh,
    const DCArrayKokkos<BoundaryConditionEnums_t>& BoundaryConditionEnums,
    const RaggedRightArrayKokkos<double>& heat_flux_bc_global_vars,
        const DCArrayKokkos<double>& bc_state_vars,
        const DCArrayKokkos<double>& q_transfer,
        const size_t bdy_node_gid,
        const size_t bdy_set, 
        const SimulationParameters_t& SimulationParamaters)
{
    // Heat flux to set the boundary to = bc_global_vars(4)
    // Set velocity to zero in the specified direction
    size_t plane = heat_flux_bc_global_vars(bdy_set, 1); // Hard coded value 1 for determining which plane (0 for x, 1 for y, 2 for z)
    size_t i, j;
    switch (static_cast<int>(plane)) {
        case 0:
            i = 1; j = 2;
            break;
        case 1:
            i = 0; j = 2;
            break;
        case 2:
            i = 0; j = 1;
            break;
        default:
            // shouldn't happen if value was validated during YAML parsing
            i = 0; j = 0;
        break;
    }

    q_transfer(bdy_node_gid) = heat_flux_bc_global_vars(bdy_set, 0) * (SimulationParamaters.MeshInput.length[i] * SimulationParamaters.MeshInput.length[j]) / (4.0 * SimulationParamaters.MeshInput.num_elems[i] * SimulationParamaters.MeshInput.num_elems[j]); // bc_global_vars(4);
    return;
} // end func

} // end namespace

#endif // end Header Guard