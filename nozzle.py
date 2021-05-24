"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

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

from pytools.obj_array import obj_array_vectorize
import pickle

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw, flatten, unflatten
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.profiling import PyOpenCLProfilingArrayContext

from mirgecom.euler import euler_operator
from mirgecom.navierstokes import ns_operator
from mirgecom.fluid import split_conserved
from mirgecom.artificial_viscosity import (
    av_operator,
    smoothness_indicator
)
from mirgecom.simutil import (
    inviscid_sim_timestep,
    sim_checkpoint,
    check_step,
    generate_and_distribute_mesh
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
# from mirgecom.checkstate import compare_states
from mirgecom.integrators import (
    rk4_step, 
    lsrk54_step, 
    lsrk144_step, 
    euler_step
)
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedInviscidBoundary,
    AdiabaticSlipBoundary,
    DummyBoundary,
    PrescribedViscousBoundary,
    IsothermalNoSlipBoundary
)
from mirgecom.initializers import (
    Lump,
    Uniform,
    PlanarDiscontinuity
)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport

from logpyle import IntervalTimer

from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (initialize_logmgr,
    logmgr_add_many_discretization_quantities, logmgr_add_cl_device_info)
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
    from meshmode.mesh.io import (
        read_gmsh,
        generate_gmsh,
        ScriptWithFilesSource
    )
    import os
    if os.path.exists("data/pseudoY1nozzle.msh") is False:
        mesh = generate_gmsh(
            ScriptWithFilesSource("""
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
        """, ["data/pseudoY1nozzle.brep"]), 3, target_unit="MM")
    else:
        mesh = read_gmsh("data/pseudoY1nozzle.msh")

    return mesh


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, casename="nozzle", user_input_file="",
         snapshot_pattern="{casename}-{step:06d}-{rank:04d}.pkl", 
         restart_step=None, restart_name=None,
         use_profiling=False, use_logmgr=False, use_lazy_eval=False):
    """Drive the Y0 nozzle example."""

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    if restart_name is None:
      restart_name=casename

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
        mode="wo", mpi_comm=comm)

    cl_ctx = ctx_factory()
    if use_profiling:
        if use_lazy_eval:
            raise RuntimeError("Cannot run lazy with profiling.")
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        if use_lazy_eval:
            actx = PytatoArrayContext(queue)
        else:
            actx = PyOpenCLArrayContext(queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    # default input values that will be (potentially) read from input
    nviz = 100
    nrestart = 100
    current_dt = 5e-8
    t_final = 5.e-6
    order = 1
    alpha_sc = 0.5
    s0_sc = -5.0
    kappa_sc = 0.5

    if(user_input_file):
        #with open('run2_params.yaml') as f:
        if rank ==0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            input_data=None
        input_data = comm.bcast(input_data, root=0)
            #print(input_data)
        nviz = int(input_data["nviz"])
        nrestart = int(input_data["nrestart"])
        current_dt = float(input_data["current_dt"])
        t_final = float(input_data["t_final"])
        order = int(input_data["order"])

    if(rank == 0):
        print(f'Simluation control data:')
        print(f'\tnviz = {nviz}')
        print(f'\tnrestart = {nrestart}')
        print(f'\tcurrent_dt = {current_dt}')
        print(f'\tt_final = {t_final}')
        print(f'\torder = {order}')
        print(f"Shock capturing parameters: alpha {alpha_sc}, s0 {s0_sc}, kappa {kappa_sc}")

    dim = 3
    exittol = .09
    current_cfl = 1.0
    vel_init = np.zeros(shape=(dim,))
    vel_inflow = np.zeros(shape=(dim,))
    vel_outflow = np.zeros(shape=(dim,))
    current_t = 0
    constant_cfl = False
    nstatus = 10000000000
    rank = 0
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
    rho_bkrnd=1.77619667e-3
    pres_bkrnd=100
    temp_bkrnd=298
     
    # nozzle inflow #
    # 
    # stagnation tempertuare 298 K
    # stagnation pressure 1.5e Pa
    # 
    # isentropic expansion based on the area ratios between the inlet (r=13e-3m) and the throat (r=6.3e-3)
    #
    # calculate the inlet Mach number from the area ratio
    nozzleInletRadius = 13.e-3
    nozzleThroatRadius = 6.3e-3
    nozzleInletArea = math.pi*nozzleInletRadius*nozzleInletRadius
    nozzleThroatArea = math.pi*nozzleThroatRadius*nozzleThroatRadius
    inletAreaRatio = nozzleInletArea/nozzleThroatArea

    def getMachFromAreaRatio(area_ratio, gamma, mach_guess=0.01):
        error=1.e-8
        nextError=1.e8
        g=gamma
        M0=mach_guess
        while nextError > error:
            R = ((2/(g+1)+((g-1)/(g+1)*M0*M0))**(((g+1)/(2*g-2))))/M0-area_ratio
            dRdM = (2*((2/(g+1)+((g-1)/(g+1)*M0*M0))**(((g+1)/(2*g-2))))/
                   (2*g-2)*(g-1)/(2/(g+1)+((g-1)/(g+1)*M0*M0))-
                   ((2/(g+1)+((g-1)/(g+1)*M0*M0))**(((g+1)/(2*g-2))))* M0**(-2))
      
            M1=M0-R/dRdM
            nextError=abs(R)
            M0=M1

        return M1


    def getIsentropicPressure(mach, P0, gamma):
        pressure=(1.+(gamma-1.)*0.5*math.pow(mach,2))
        pressure=P0*math.pow(pressure,(-gamma/(gamma-1.)))
        return pressure

  
    def getIsentropicTemperature(mach, T0, gamma):
      temperature=(1.+(gamma-1.)*0.5*math.pow(mach,2))
      temperature=T0*math.pow(temperature,-1.0)
      return temperature


    inlet_mach = getMachFromAreaRatio(area_ratio = inletAreaRatio, gamma=gamma_CO2, mach_guess = 0.01);
    # ramp the stagnation pressure
    start_ramp_pres = 1000
    ramp_interval = 1.e-3
    t_ramp_start = 1e-5
    pres_inflow = getIsentropicPressure(mach=inlet_mach, P0=start_ramp_pres, gamma=gamma_CO2)
    temp_inflow = getIsentropicTemperature(mach=inlet_mach, T0=298, gamma=gamma_CO2)
    rho_inflow = pres_inflow/temp_inflow/R_CO2

    print(f'inlet Mach number {inlet_mach}')
    print(f'inlet temperature {temp_inflow}')
    print(f'inlet pressure {pres_inflow}')

    end_ramp_pres = 150000
    pres_inflow_final = getIsentropicPressure(mach=inlet_mach, P0=end_ramp_pres, gamma=gamma_CO2)

    print(f'final inlet pressure {pres_inflow_final}')

    vel_inflow[0] = inlet_mach*math.sqrt(gamma_CO2*pres_inflow/rho_inflow)

    # starting pressure for the inflow ramp

    timestepper = rk4_step
    #timestepper = lsrk54_step
    #timestepper = lsrk144_step
    #timestepper = euler_step
    mu = 1.e-5
    kappa = rho_bkrnd*mu/0.75
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)
    eos = IdealSingleGas(gamma=gamma_CO2, gas_const=R_CO2, transport_model=transport_model)
    bulk_init = PlanarDiscontinuity(dim=dim, disc_location=-.30, sigma=0.005,
                              temperature_left=temp_inflow, temperature_right=temp_bkrnd,
                              pressure_left=pres_inflow, pressure_right=pres_bkrnd,
                              velocity_left=vel_inflow, velocity_right=vel_outflow)

    # pressure ramp function
    def inflow_ramp_pressure(t, startP=start_ramp_pres, finalP=end_ramp_pres, 
                             ramp_interval=ramp_interval, t_ramp_start=t_ramp_start):
      if t > t_ramp_start:
          rampPressure = min(finalP, startP+(t-t_ramp_start)/ramp_interval*(finalP-startP))
      else:
          rampPressure = startP
      return rampPressure


    class IsentropicInflow:

        def __init__(self, *, dim=1, direc=0, T0=298, P0=1e5, mach= 0.01, p_fun = None):

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
            pressure = getIsentropicPressure(mach=self._mach, P0=P0, gamma=gamma)
            temperature = getIsentropicTemperature(mach=self._mach, T0=T0, gamma=gamma)
            rho = pressure/temperature/gas_const

            #print(f'ramp Mach number {self._mach}')
            #print(f'ramp stagnation pressure {P0}')
            #print(f'ramp stagnation temperature {T0}')
            #print(f'ramp pressure {pressure}')
            #print(f'ramp temperature {temperature}')

            velocity = np.zeros(shape=(self._dim,)) 
            velocity[self._direc] = self._mach*math.sqrt(gamma*pressure/rho)
    
            mass = 0.0*x_vec[0] + rho
            mom = velocity*mass
            energy = (pressure/(gamma - 1.0)) + np.dot(mom, mom)/(2.0*mass)
            from mirgecom.fluid import join_conserved
            return join_conserved(dim=self._dim, mass=mass, momentum=mom, energy=energy)


    inflow_init = IsentropicInflow(dim=dim, T0=298, P0=start_ramp_pres, 
                                   mach = inlet_mach , p_fun=inflow_ramp_pressure)
    outflow_init = Uniform(dim=dim, rho=rho_bkrnd, p=pres_bkrnd,
                           velocity=vel_outflow)

    #inflow = PrescribedViscousBoundary(q_func=inflow_init)
    #outflow = PrescribedViscousBoundary(q_func=outflow_init)
    inflow = PrescribedInviscidBoundary(fluid_solution_func=inflow_init)
    outflow = PrescribedInviscidBoundary(fluid_solution_func=outflow_init)
    wall = IsothermalNoSlipBoundary()


    # timestep estimate
    #wave_speed = max(mach2*c_bkrnd,c_shkd+velocity2[0])
    #char_len = 0.001
    #area=char_len*char_len/2
    #perimeter = 2*char_len+math.sqrt(2*char_len*char_len)
    #h = 2*area/perimeter

    #dt_est = 1/(wave_speed*order*order/h)
    #print(f"Time step estimate {dt_est}\n")
#
    #dt_est_visc = 1/(wave_speed*order*order/h+alpha_sc*order*order*order*order/h/h)
    #print(f"Viscous timestep estimate {dt_est_visc}\n")

    boundaries = {
        DTAG_BOUNDARY("Inflow"): inflow,
        DTAG_BOUNDARY("Outflow"): outflow,
        DTAG_BOUNDARY("Wall"): wall
    }

    if restart_step is None:
        local_mesh, global_nelements = generate_and_distribute_mesh(comm, get_pseudo_y0_mesh)
        local_nelements = local_mesh.nelements

    else:  # Restart
        with open('restart_data/'+snapshot_pattern.format(casename=restart_name, step=restart_step, rank=rank), "rb") as f:
            restart_data = pickle.load(f)

        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]

        assert comm.Get_size() == restart_data["num_parts"]

    if rank == 0:
        logging.info("Making discretization")
    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    # initialize the sponge field
    def gen_sponge():
        thickness = 0.15
        amplitude = 1./current_dt/25.
        x0 = 0.05
        
        return (amplitude *  actx.np.where(
            nodes[0]>x0,
            zeros+((nodes[0]-x0)/thickness)*((nodes[0]-x0)/thickness),
            zeros+0.0))

    zeros = 0 * nodes[0]
    sponge_sigma =  gen_sponge()
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

        current_state = unflatten(
            actx, discr.discr_from_dd("vol"),
            obj_array_vectorize(actx.from_numpy, restart_data["state"]))

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
            extract_vars_for_logging, units_for_logging)
        #logmgr_add_package_versions(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max", "t_log.max",
                            "min_pressure", "max_pressure",
                            "min_temperature", "max_temperature"])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    #visualizer = make_visualizer(discr, order + 3
                                 #if discr.dim == 2 else order)
    visualizer = make_visualizer(discr)

    initname = "pseudoY0"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    get_timestep = partial(inviscid_sim_timestep, discr=discr, t=current_t,
                           dt=current_dt, cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    
    def sponge(q, q_ref, sigma):
        return(sigma*(q_ref-q))

    def my_rhs(t, state):

        # check for some troublesome output types
        inf_exists = not np.isfinite(discr.norm(state, np.inf))
        if inf_exists:
            if rank == 0:
                logging.info("Non-finite values detected in simulation, exiting...")
            # dump right now
            cv = split_conserved(dim, state)
            tagged_cells = smoothness_indicator(discr, cv.mass, s0=s0_sc, kappa=kappa_sc)
            viz_fields = [("sponge_sigma", gen_sponge()),("tagged cells", tagged_cells)]
            sim_checkpoint(discr=discr, visualizer=visualizer, eos=eos,
                              q=state, vizname=casename,
                              step=999999999, t=t, dt=current_dt,
                              nviz=1, exittol=exittol,
                              constant_cfl=constant_cfl, comm=comm, vis_timer=vis_timer,
                              overwrite=True, viz_fields=viz_fields)
            exit()

        #return ( euler_operator(discr, q=state, t=t,boundaries=boundaries, eos=eos)
               #+ av_operator(discr,t=t, q=state, eos=eos, boundaries=boundaries,
                 #alpha=alpha_sc, s0=s0_sc, kappa=kappa_sc)
               #+ sponge(q=state, q_ref=ref_state, sigma=sponge_sigma))
        return ( 
            ns_operator(discr, q=state, t=t, boundaries=boundaries, eos=eos) +
            av_operator(
                discr, q=state, boundaries=boundaries,
                boundary_kwargs={"time": t, "eos":eos},
                alpha=alpha_sc, s0=s0_sc, kappa=kappa_sc
            ) +
            sponge(q=state, q_ref=ref_state, sigma=sponge_sigma)
        )


    restart_path='restart_data/'
    viz_path='viz_data/'
    if(rank == 0):
        if not os.path.exists(restart_path):
            os.makedirs(restart_path)  
        if not os.path.exists(viz_path):
            os.makedirs(viz_path)  

    def my_checkpoint(step, t, dt, state):

        write_restart = (check_step(step, nrestart)
                         if step != restart_step else False)
        if write_restart is True:
            with open(restart_path+snapshot_pattern.format(casename=casename, step=step, rank=rank), "wb") as f:
                pickle.dump({
                    "local_mesh": local_mesh,
                    "state": obj_array_vectorize(actx.to_numpy, flatten(state)),
                    "t": t,
                    "step": step,
                    "global_nelements": global_nelements,
                    "num_parts": nparts,
                    }, f)

        cv = split_conserved(dim, state)
        tagged_cells = smoothness_indicator(discr, cv.mass, s0=s0_sc, kappa=kappa_sc)
        viz_fields = [("sponge_sigma", gen_sponge()),("tagged cells", tagged_cells)]
        return sim_checkpoint(discr=discr, visualizer=visualizer, eos=eos,
                              q=state, vizname=viz_path+casename,
                              step=step, t=t, dt=dt, nstatus=nstatus,
                              nviz=nviz, exittol=exittol,
                              constant_cfl=constant_cfl, comm=comm, vis_timer=vis_timer,
                              overwrite=True, viz_fields=viz_fields)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, current_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      checkpoint=my_checkpoint,
                      get_timestep=get_timestep, state=current_state,
                      t_final=t_final, t=current_t, istep=current_step,
                      logmgr=logmgr,eos=eos,dim=dim)

    if rank == 0:
        logger.info("Checkpointing final state ...")

    my_checkpoint(current_step, t=current_t,
                  dt=(current_t - checkpoint_t),
                  state=current_state)

    if current_t - t_final < 0:
        raise ValueError("Simulation exited abnormally")

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument('-r', '--restart_file',  type=ascii, 
                        dest='restart_file', nargs='?', action='store', 
                        help='simulation restart file')
    parser.add_argument('-i', '--input_file',  type=ascii,
                        dest='input_file', nargs='?', action='store',
                        help='simulation config file')
    parser.add_argument('-c', '--casename',  type=ascii,
                        dest='casename', nargs='?', action='store',
                        help='simulation case name')
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "nozzle"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'","")
    else:
        print(f"Default casename {casename}")

    snapshot_pattern="{casename}-{step:06d}-{rank:04d}.pkl"
    restart_step=None
    restart_name=None
    if(args.restart_file):
        print(f"Restarting from file {args.restart_file}")
        file_path, file_name = os.path.split(args.restart_file)
        restart_step = int(file_name.split('-')[1])
        restart_name = (file_name.split('-')[0]).replace("'","")
        print(f"step {restart_step}")
        print(f"name {restart_name}") 

    if(args.input_file):
        input_file = (args.input_file).replace("'","")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_step=restart_step, restart_name=restart_name, user_input_file=input_file,
         snapshot_pattern=snapshot_pattern,
         use_profiling=args.profile, use_lazy_eval=args.lazy, use_logmgr=args.log)

# vim: foldmethod=marker
