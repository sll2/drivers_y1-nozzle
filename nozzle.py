"""Run the nozzle case."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import os
import yaml
import logging
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial
import math

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.op import nodal_max

from mirgecom.profiling import PyOpenCLProfilingArrayContext

from mirgecom.navierstokes import ns_operator
from mirgecom.fluid import make_conserved
from mirgecom.artificial_viscosity import (av_operator, smoothness_indicator)
from mirgecom.inviscid import get_inviscid_cfl
from mirgecom.simutil import (
    inviscid_sim_timestep,
    check_step,
    generate_and_distribute_mesh, write_visfile,
    check_naninf_local,
    check_range_local
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
# from mirgecom.checkstate import compare_states
from mirgecom.integrators import (rk4_step, lsrk54_step, lsrk144_step,
                                  euler_step)
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedInviscidBoundary,
    IsothermalNoSlipBoundary
)
from mirgecom.initializers import (Uniform, PlanarDiscontinuity)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport

from logpyle import IntervalTimer

from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr, logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info, logmgr_set_time, LogUserQuantity
)

logger = logging.getLogger(__name__)


def get_pseudo_y0_mesh():
    """Generate or import a grid using `gmsh`.

    Input required:
        data/pseudoY0.brep  (for mesh gen)
        -or-
        data/pseudoY0.msh   (read existing mesh)

    This routine will generate a new grid if it does
    not find the grid file (data/pseudoY0.msh), but
    note that if the grid is generated in millimeters,
    then the solution initialization and BCs need to be
    adjusted or the grid needs to be scaled up to meters
    before being used with the current main driver in this
    example.
    """
    from meshmode.mesh.io import (read_gmsh, generate_gmsh,
                                  ScriptWithFilesSource)
    import os
    if os.path.exists("data/pseudoY1nozzle.msh") is False:
        mesh = generate_gmsh(ScriptWithFilesSource(
            """
            Merge "data/pseudoY1nozzle.brep";
            Mesh.CharacteristicLengthMin = 1;
            Mesh.CharacteristicLengthMax = 10;
            Mesh.ElementOrder = 2;
            Mesh.CharacteristicLengthExtendFromBoundary = 0;

            // Inside and end surfaces of nozzle/scramjet
            Field[1] = Distance;
            Field[1].NNodesByEdge = 100;
            Field[1].FacesList = {5,7,8,9,10};
            Field[2] = Threshold;
            Field[2].IField = 1;
            Field[2].LcMin = 1;
            Field[2].LcMax = 10;
            Field[2].DistMin = 0;
            Field[2].DistMax = 20;

            // Edges separating surfaces with boundary layer
            // refinement from those without
            // (Seems to give a smoother transition)
            Field[3] = Distance;
            Field[3].NNodesByEdge = 100;
            Field[3].EdgesList = {5,10,14,16};
            Field[4] = Threshold;
            Field[4].IField = 3;
            Field[4].LcMin = 1;
            Field[4].LcMax = 10;
            Field[4].DistMin = 0;
            Field[4].DistMax = 20;

            // Min of the two sections above
            Field[5] = Min;
            Field[5].FieldsList = {2,4};

            Background Field = 5;
        """, ["data/pseudoY1nozzle.brep"]),
                             3,
                             target_unit="MM")
    else:
        mesh = read_gmsh("data/pseudoY1nozzle.msh")

    return mesh


@mpi_entry_point
def main(ctx_factory=cl.create_some_context,
         casename="nozzle",
         user_input_file=None,
         snapshot_pattern="{casename}-{step:06d}-{rank:04d}.pkl",
         restart_step=None,
         restart_name=None,
         use_profiling=False,
         use_logmgr=False,
         use_lazy_eval=False):
    """Drive the Y0 nozzle example."""

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    if restart_name is None:
        restart_name = casename

    logmgr = initialize_logmgr(use_logmgr,
                               filename=(f"{casename}.sqlite"),
                               mode="wo",
                               mpi_comm=comm)

    cl_ctx = ctx_factory()
    if use_profiling:
        if use_lazy_eval:
            raise RuntimeError("Cannot run lazy with profiling.")
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(
            queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        if use_lazy_eval:
            from meshmode.array_context import PytatoPyOpenCLArrayContext
            actx = PytatoPyOpenCLArrayContext(queue)
        else:
            actx = PyOpenCLArrayContext(
                queue,
                allocator=cl_tools.MemoryPool(
                    cl_tools.ImmediateAllocator(queue)))

    # default input values that will be read from input (if they exist)
    nviz = 100
    nrestart = 100
    nhealth = 100
    nstatus = 1
    current_dt = 5e-8
    t_final = 5.0e-6
    order = 1
    alpha_sc = 0.5
    s0_sc = -5.0
    kappa_sc = 0.5
    integrator = "rk4"

    if user_input_file:
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            input_data = None
        input_data = comm.bcast(input_data, root=0)
        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            alpha_sc = float(input_data["alpha_sc"])
        except KeyError:
            pass
        try:
            kappa_sc = float(input_data["kappa_sc"])
        except KeyError:
            pass
        try:
            s0_sc = float(input_data["s0_sc"])
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if rank == 0:
        print("#### Simluation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tShock capturing parameters: alpha {alpha_sc}, "
              f"s0 {s0_sc}, kappa {kappa_sc}")
        print(f"\tTime integration {integrator}")
        print("#### Simluation control data: ####")

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step
    restart_path = "restart_data/"
    viz_path = "viz_data/"

    dim = 3
    current_cfl = 1.0
    vel_inflow = np.zeros(shape=(dim, ))
    vel_outflow = np.zeros(shape=(dim, ))
    current_t = 0
    constant_cfl = False
    checkpoint_t = current_t
    current_step = 0

    # working gas: CO2 #
    #   gamma = 1.289
    #   MW=44.009  g/mol
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma_CO2 = 1.289
    R_CO2 = 8314.59/44.009

    # background
    #   100 Pa
    #   298 K
    #   rho = 1.77619667e-3 kg/m^3
    #   velocity = 0,0,0
    rho_bkrnd = 1.77619667e-3
    pres_bkrnd = 100
    temp_bkrnd = 298

    # nozzle inflow #
    #
    # stagnation tempertuare 298 K
    # stagnation pressure 1.5e Pa
    #
    # isentropic expansion based on the area ratios between the inlet (r=13e-3m)
    # and the throat (r=6.3e-3)
    #
    # calculate the inlet Mach number from the area ratio
    nozzle_inlet_radius = 13.0e-3
    nozzle_throat_radius = 6.3e-3
    nozzle_inlet_area = math.pi*nozzle_inlet_radius*nozzle_inlet_radius
    nozzle_throat_area = math.pi*nozzle_throat_radius*nozzle_throat_radius
    inlet_area_ratio = nozzle_inlet_area/nozzle_throat_area

    def getMachFromAreaRatio(area_ratio, gamma, mach_guess=0.01):
        error = 1.0e-8
        nextError = 1.0e8
        g = gamma
        M0 = mach_guess
        while nextError > error:
            R = (((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))/M0
                - area_ratio)
            dRdM = (2*((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
                   / (2*g - 2)*(g - 1)/(2/(g + 1) + ((g - 1)/(g + 1)*M0*M0)) -
                   ((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
                   * M0**(-2))
            M1 = M0 - R/dRdM
            nextError = abs(R)
            M0 = M1

        return M1

    def getIsentropicPressure(mach, P0, gamma):
        pressure = (1. + (gamma - 1.)*0.5*math.pow(mach, 2))
        pressure = P0*math.pow(pressure, (-gamma / (gamma - 1.)))
        return pressure

    def getIsentropicTemperature(mach, T0, gamma):
        temperature = (1. + (gamma - 1.)*0.5*math.pow(mach, 2))
        temperature = T0*math.pow(temperature, -1.0)
        return temperature

    inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                      gamma=gamma_CO2,
                                      mach_guess=0.01)
    # ramp the stagnation pressure
    start_ramp_pres = 1000
    ramp_interval = 1.0e-3
    t_ramp_start = 1.0e-5
    pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                        P0=start_ramp_pres,
                                        gamma=gamma_CO2)
    temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                           T0=298,
                                           gamma=gamma_CO2)
    rho_inflow = pres_inflow / temp_inflow / R_CO2
    end_ramp_pres = 150000
    pres_inflow_final = getIsentropicPressure(mach=inlet_mach,
                                              P0=end_ramp_pres,
                                              gamma=gamma_CO2)
    vel_inflow[0] = inlet_mach * math.sqrt(
        gamma_CO2 * pres_inflow / rho_inflow)

    if rank == 0:
        print(f"inlet Mach number {inlet_mach}")
        print(f"inlet temperature {temp_inflow}")
        print(f"inlet pressure {pres_inflow}")
        print(f"final inlet pressure {pres_inflow_final}")

    mu = 1.e-5
    kappa = rho_bkrnd*mu/0.75
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)
    eos = IdealSingleGas(
        gamma=gamma_CO2,
        gas_const=R_CO2,
        transport_model=transport_model
    )
    bulk_init = PlanarDiscontinuity(dim=dim, disc_location=-.30, sigma=0.005,
        temperature_left=temp_inflow, temperature_right=temp_bkrnd,
        pressure_left=pres_inflow, pressure_right=pres_bkrnd,
        velocity_left=vel_inflow, velocity_right=vel_outflow)

    # pressure ramp function
    def inflow_ramp_pressure(
        t,
        startP=start_ramp_pres,
        finalP=end_ramp_pres,
        ramp_interval=ramp_interval,
        t_ramp_start=t_ramp_start
    ):
        if t > t_ramp_start:
            rampPressure = min(
                finalP, startP + (t - t_ramp_start)/ramp_interval *
                (finalP - startP)
            )
        else:
            rampPressure = startP
        return rampPressure

    class IsentropicInflow:
        def __init__(self, *, dim=1, direc=0, T0=298, P0=1e5, mach=0.01, p_fun=None):

            self._P0 = P0
            self._T0 = T0
            self._dim = dim
            self._direc = direc
            self._mach = mach
            if p_fun is not None:
                self._p_fun = p_fun

        def __call__(self, x_vec, *, time=0, eos, **kwargs):

            if self._p_fun is not None:
                P0 = self._p_fun(time)
            else:
                P0 = self._P0
            T0 = self._T0

            gamma = eos.gamma()
            gas_const = eos.gas_const()
            pressure = getIsentropicPressure(
                mach=self._mach,
                P0=P0,
                gamma=gamma
            )
            temperature = getIsentropicTemperature(
                mach=self._mach,
                T0=T0,
                gamma=gamma
            )
            rho = pressure/temperature/gas_const

            velocity = np.zeros(shape=(self._dim, ))
            velocity[self._direc] = self._mach*math.sqrt(gamma*pressure/rho)

            mass = 0.0*x_vec[0] + rho
            mom = velocity*mass
            energy = (pressure/(gamma - 1.0)) + np.dot(mom, mom)/(2.0*mass)
            return make_conserved(
                dim=self._dim,
                mass=mass,
                momentum=mom,
                energy=energy
            )

    inflow_init = IsentropicInflow(
        dim=dim,
        T0=298,
        P0=start_ramp_pres,
        mach=inlet_mach,
        p_fun=inflow_ramp_pressure
    )
    outflow_init = Uniform(
        dim=dim,
        rho=rho_bkrnd,
        p=pres_bkrnd,
        velocity=vel_outflow
    )

    #inflow = PrescribedViscousBoundary(q_func=inflow_init)
    #outflow = PrescribedViscousBoundary(q_func=outflow_init)
    inflow = PrescribedInviscidBoundary(fluid_solution_func=inflow_init)
    outflow = PrescribedInviscidBoundary(fluid_solution_func=outflow_init)
    wall = IsothermalNoSlipBoundary()

    boundaries = {
        DTAG_BOUNDARY("Inflow"): inflow,
        DTAG_BOUNDARY("Outflow"): outflow,
        DTAG_BOUNDARY("Wall"): wall
    }

    if restart_step is None:
        local_mesh, global_nelements = generate_and_distribute_mesh(
            comm,
            get_pseudo_y0_mesh
        )
        local_nelements = local_mesh.nelements

    else:  # Restart

        from mirgecom.restart import read_restart_data
        restart_file = "restart_data/"+snapshot_pattern.format(casename=restart_name,
                                                               step=restart_step,
                                                               rank=rank)
        restart_data = read_restart_data(actx, restart_file)

        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]

        assert comm.Get_size() == restart_data["num_parts"]

    if rank == 0:
        logging.info("Making discretization")

    discr = EagerDGDiscretization(actx,
                                  local_mesh,
                                  order=order,
                                  mpi_communicator=comm)

    nodes = thaw(actx, discr.nodes())

    # initialize the sponge field
    def gen_sponge():
        thickness = 0.15
        amplitude = 1.0/current_dt/25.0
        x0 = 0.05

        return (amplitude * actx.np.where(
            nodes[0] > x0, zeros + ((nodes[0] - x0) / thickness) *
            ((nodes[0] - x0) / thickness), zeros + 0.0))

    zeros = 0 * nodes[0]
    sponge_sigma = gen_sponge()
    ref_state = bulk_init(x_vec=nodes, eos=eos, time=0.0)

    if restart_step is None:
        if rank == 0:
            logging.info("Initializing soln.")
        # for Discontinuity initial conditions
        current_state = bulk_init(x_vec=nodes, eos=eos, time=0.0)
        # for uniform background initial condition
        #current_state = bulk_init(nodes, eos=eos)
    else:
        current_t = restart_data["t"]
        current_step = restart_step
        current_state = restart_data["state"]

    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                                                  extract_vars_for_logging,
                                                  units_for_logging)
        logmgr_set_time(logmgr, current_step, current_t)
        logmgr.add_quantity(log_cfl, interval=nstatus)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("cfl.max", "cfl = {value:1.4f}\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure", "{value:1.9e})\n"),
            ("min_temperature", "------- T (min, max) (K)  = ({value:7g}, "),
            ("max_temperature", "{value:7g})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(discr)

    initname = "pseudoY0"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim,
                                     order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt,
                                     t_final=t_final,
                                     nstatus=nstatus,
                                     nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname,
                                     casename=casename)
    if rank == 0:
        logger.info(init_message)

    get_timestep = partial(inviscid_sim_timestep,
                           discr=discr,
                           t=current_t,
                           dt=current_dt,
                           cfl=current_cfl,
                           eos=eos,
                           t_final=t_final,
                           constant_cfl=constant_cfl)

    def sponge(cv, cv_ref, sigma):
        return (sigma*(cv_ref - cv))

    def my_rhs(t, state):
        return (
            ns_operator(discr, cv=state, t=t, boundaries=boundaries, eos=eos) +
            make_conserved(dim,
                           q=av_operator(discr,
                                         q=state.join(),
                                         boundaries=boundaries,
                                         boundary_kwargs={
                                             "time": t,
                                             "eos": eos
                                         },
                                         alpha=alpha_sc,
                                         s0=s0_sc,
                                         kappa=kappa_sc)) +
            sponge(cv=state, cv_ref=ref_state, sigma=sponge_sigma))

    def my_checkpoint(step, t, dt, state, force=False):
        do_health = force or check_step(step, nhealth) and step > 0
        do_viz = force or check_step(step, nviz)
        do_restart = force or check_step(step, nrestart)
        do_status = force or check_step(step, nstatus)

        if do_viz or do_health:
            dv = eos.dependent_vars(state)

        errors = False
        if do_health:
            health_message = ""
            if check_naninf_local(discr, "vol", dv.pressure):
                errors = True
                health_message += "Invalid pressure data found.\n"
            elif check_range_local(discr,
                                   "vol",
                                   dv.pressure,
                                   min_value=1,
                                   max_value=2.0e6):
                errors = True
                health_message += "Pressure data failed health check.\n"

        errors = comm.allreduce(errors, MPI.LOR)
        if errors:
            if rank == 0:
                logger.info("Fluid solution failed health check.")
            if health_message:
                logger.info(f"{rank=}:  {health_message}")

        #if check_step(step, nrestart) and step != restart_step and not errors:
        if do_restart or errors:
            filename = restart_path + snapshot_pattern.format(
                step=step, rank=rank, casename=casename)
            restart_dictionary = {
                "local_mesh": local_mesh,
                "order": order,
                "state": state,
                "t": t,
                "step": step,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, restart_dictionary, filename, comm)

        if do_status or do_viz or errors:
            local_cfl = get_inviscid_cfl(discr, eos=eos, dt=dt, cv=state)
            max_cfl = nodal_max(discr, "vol", local_cfl)
            log_cfl.set_quantity(max_cfl)

        #if ((check_step(step, nviz) and step != restart_step) or errors):
        if do_viz or errors:
            tagged_cells = smoothness_indicator(discr,
                                                state.mass,
                                                s0=s0_sc,
                                                kappa=kappa_sc)
            viz_fields = [("cv", state), ("dv", eos.dependent_vars(state)),
                          ("sponge_sigma", gen_sponge()),
                          ("tagged_cells", tagged_cells), ("cfl", local_cfl)]
            write_visfile(discr,
                          viz_fields,
                          visualizer,
                          vizname=viz_path + casename,
                          step=step,
                          t=t,
                          overwrite=True,
                          vis_timer=vis_timer)

        if errors:
            raise RuntimeError("Error detected by user checkpoint, exiting.")

        return dt

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, current_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      checkpoint=my_checkpoint,
                      get_timestep=get_timestep, state=current_state,
                      t_final=t_final, t=current_t, istep=current_step,
                      logmgr=logmgr, eos=eos, dim=dim)

    if rank == 0:
        logger.info("Checkpointing final state ...")
    my_checkpoint(current_step,
                  t=current_t,
                  dt=(current_t - checkpoint_t),
                  state=current_state,
                  force=True)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r",
                        "--restart_file",
                        type=ascii,
                        dest="restart_file",
                        nargs="?",
                        action="store",
                        help="simulation restart file")
    parser.add_argument("-i",
                        "--input_file",
                        type=ascii,
                        dest="input_file",
                        nargs="?",
                        action="store",
                        help="simulation config file")
    parser.add_argument("-c",
                        "--casename",
                        type=ascii,
                        dest="casename",
                        nargs="?",
                        action="store",
                        help="simulation case name")
    parser.add_argument("--profile",
                        action="store_true",
                        default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--log",
                        action="store_true",
                        default=True,
                        help="enable logging profiling [ON]")
    parser.add_argument("--lazy",
                        action="store_true",
                        default=False,
                        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "nozzle"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")

    snapshot_pattern = "{casename}-{step:06d}-{rank:04d}.pkl"
    restart_step = None
    restart_name = None
    if args.restart_file:
        print(f"Restarting from file {args.restart_file}")
        file_path, file_name = os.path.split(args.restart_file)
        restart_step = int(file_name.split("-")[1])
        restart_name = (file_name.split("-")[0]).replace("'", "")
        print(f"step {restart_step}")
        print(f"name {restart_name}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_step=restart_step,
         restart_name=restart_name,
         user_input_file=input_file,
         snapshot_pattern=snapshot_pattern,
         use_profiling=args.profile,
         use_lazy_eval=args.lazy,
         use_logmgr=args.log)

# vim: foldmethod=marker
