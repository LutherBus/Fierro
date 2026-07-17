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
        const size_t bdy_set)
{
    // Heat flux to set the boundary to = bc_global_vars(4)
    // Set velocity to zero in the specified direction
    q_transfer(bdy_node_gid) = heat_flux_bc_global_vars(bdy_set,0); // bc_global_vars(4);

    return;
} // end func

} // end namespace

#endif // end Header Guard