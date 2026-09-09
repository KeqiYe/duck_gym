"""Run physical regression cases and measure both engines under explicit conditions."""
import argparse
from datetime import datetime
import csv
import json
import hashlib
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import mujoco
from model_io import ROOT, export_model, prepare_microduck, mul


def scene(body, ground=False, gravity='0 0 -9.81'):
    floor = '<geom name="floor" type="plane" size="3 3 .1" friction=".8 0 0" contype="2" conaffinity="1"/>' if ground else ''
    return f'''<mujoco><compiler angle="radian"/><option timestep=".001" gravity="{gravity}" integrator="Euler" iterations="100" tolerance="1e-10" cone="elliptic"/><default><geom friction=".8 0 0" contype="1" conaffinity="2" solref=".002 1" solimp=".99 .99 .001"/></default><worldbody>{floor}{body}</worldbody></mujoco>'''


def cases():
    ball = '<body name="ball" pos="0 0 1"><freejoint/><geom type="sphere" size=".05" mass="1"/></body>'
    pendulum = '<body name="pendulum" pos="0 0 1"><joint name="hinge" type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 -.5" size=".03" mass="1"/></body>'
    motor = '<body name="rotor"><joint name="hinge" type="hinge" axis="0 0 1" damping=".01" armature=".002"/><geom type="box" size=".1 .05 .02" mass="1"/></body>'
    yield 'freefall', scene(ball), {}, {}, 0.2, 1e-8
    yield 'pendulum', scene(pendulum), {'hinge': .5}, {}, 0.5, 0.01
    yield 'motor', scene(motor, gravity='0 0 0'), {}, {'hinge': .01}, 0.5, 0.01
    dry = motor.replace('damping=".01"', 'damping=".01" frictionloss=".0048"')
    yield 'joint_friction', scene(dry, gravity='0 0 0'), {}, {'hinge': .01}, .5, .01
    yield 'joint_stiction', scene(dry, gravity='0 0 0'), {}, {'hinge': .002}, .5, .001
    limit = motor.replace('damping=".01" armature=".002"', 'range="-.2 .2" limited="true"')
    yield 'joint_limit', scene(limit, gravity='0 0 0'), {}, {'hinge': .02}, 0.5, 0.02
    yield 'sphere_drop', scene(ball, ground=True), {}, {}, 0.8, 0.03
    box = '<body name="box" pos="0 0 .08"><freejoint/><geom type="box" size=".1 .08 .08" mass="1"/></body>'
    yield 'box_rest', scene(box, ground=True), {}, {}, 0.5, .003
    yield 'box_slide', scene(box, ground=True), {'vx': .5}, {}, 0.5, .02
    spin = '<body name="spinner" pos="0 0 1"><freejoint/><geom type="box" size=".1 .07 .04" mass="1"/></body>'
    yield 'free_rotation', scene(spin, gravity='0 0 0'), {'omega': [1,2,3]}, {}, .2, .01


def run_case(name, xml, initial, torques, duration, tolerance, args):
    folder = args.output / name; folder.mkdir(parents=True, exist_ok=True)
    (folder / 'scene.xml').write_text(xml)
    m = mujoco.MjModel.from_xml_string(xml); m.opt.timestep = args.dt
    d = mujoco.MjData(m)
    for key, value in initial.items():
        if key == 'vx': d.qvel[0] = value
        elif key == 'omega': d.qvel[3:6] = value
        else: d.qpos[m.joint(key).qposadr[0]] = value
    for key, value in torques.items():
        d.qfrc_applied[m.joint(key).dofadr[0]] = value
    mujoco.mj_forward(m, d)
    export_model(m, d, folder / 'model.duck', torques)
    steps = round(duration / args.dt)
    command = [str(args.binary), str(folder/'model.duck'), '--steps', str(steps), '--dt', str(args.dt), '--iterations', str(args.iterations), '--output', str(folder/'cpu.csv')]
    proc = subprocess.run(command, check=True, text=True, capture_output=True)
    stats = json.loads(proc.stdout)
    cpu = np.genfromtxt(folder/'cpu.csv', delimiter=',', names=True)
    cpu_positions = np.column_stack([cpu[k] for k in ('x','y','z')]).reshape(steps+1, m.nbody-1, 3)
    cpu_quats = np.column_stack([cpu[k] for k in ('qw','qx','qy','qz')]).reshape(steps+1, m.nbody-1, 4)
    positions=[]; quats=[]
    def sample():
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        positions.append(d.xipos[1:].copy())
        quats.append(np.array([mul(d.xquat[i],m.body_iquat[i]) for i in range(1,m.nbody)]))
    initial_qpos, initial_qvel = d.qpos.copy(), d.qvel.copy()
    sample()
    for _ in range(steps):
        mujoco.mj_step(m,d)
        sample()
    positions, quats = np.array(positions), np.array(quats)
    np.savez_compressed(folder/'mujoco.npz', position=positions, quaternion=quats)
    position_error = float(np.linalg.norm(cpu_positions-positions, axis=-1).max())
    dots = np.clip(np.abs(np.sum(cpu_quats*quats, axis=-1)), 0, 1)
    orientation_error = float((2*np.arccos(dots)).max())
    # Separate timing: native multi-step call without Python logging or trajectory copies.
    timings=[]
    for _ in range(3):
        mujoco.mj_resetData(m,d);d.qpos[:]=initial_qpos;d.qvel[:]=initial_qvel
        for key,value in torques.items():d.qfrc_applied[m.joint(key).dofadr[0]]=value
        mujoco.mj_forward(m,d)
        t=time.perf_counter();mujoco.mj_step(m,d,nstep=steps);timings.append(time.perf_counter()-t)
    cpu_timings=[]
    for _ in range(3):
        result=subprocess.run(command[:-2],check=True,text=True,capture_output=True)
        cpu_timings.append(json.loads(result.stdout)['seconds'])
    stats.update(name=name, num_envs=1, max_position_error_m=position_error, max_orientation_error_rad=orientation_error, position_tolerance_m=tolerance, orientation_tolerance_rad=0.05, cpu_steps_per_second=steps/float(np.median(cpu_timings)), mujoco_steps_per_second=steps/float(np.median(timings)), cpu_timing_seconds=cpu_timings, mujoco_timing_seconds=timings, command=command)
    stats['reference_constraints'] = 'upstream soft constraints' if name == 'microduck_original_softness' else 'solref=0.002, solimp=0.99 for contact; see saved scene.xml'
    stats['passed'] = position_error <= tolerance and orientation_error <= .05 and stats['joint_error_m'] < .001 and stats['axis_error'] < .001 and stats['penetration_m'] < .005
    if name == 'freefall':
        t=np.arange(steps+1)*args.dt
        expected=1-9.81*args.dt**2*np.arange(steps+1)*(np.arange(steps+1)+1)/2
        stats['analytic_z_error_m']=float(np.max(np.abs(cpu_positions[:,0,2]-expected)))
        stats['passed'] &= stats['analytic_z_error_m'] < 1e-8
    (folder/'report.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(f"{name}: position={position_error:.3g} m, orientation={orientation_error:.3g} rad, joints={stats['joint_error_m']:.3g} m, CPU={stats['cpu_steps_per_second']:.0f} / MuJoCo={stats['mujoco_steps_per_second']:.0f} steps/s, passed={stats['passed']}",flush=True)
    return stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'runs'/('validation-'+datetime.now().strftime('%Y%m%d-%H%M%S')))
    parser.add_argument('--binary',type=Path,default=ROOT/'build/cpu-release/duck_sim')
    parser.add_argument('--dt',type=float,default=.001)
    parser.add_argument('--iterations',type=int,default=200)
    parser.add_argument('--case',action='append')
    parser.add_argument('--microduck',action='store_true')
    parser.add_argument('--original-softness',action='store_true',help='Add a diagnostic against upstream soft constraints; its tolerance may fail')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    selected=list(cases())
    if args.microduck:selected.append(('microduck_release',prepare_microduck(),{}, {}, 1.0, .02))
    if args.original_softness:selected.append(('microduck_original_softness',prepare_microduck(stiff_reference=False),{}, {},1.0,.02))
    reports=[]
    for case in selected:
        if args.case and case[0] not in args.case:continue
        reports.append(run_case(*case,args))
    if not reports:raise SystemExit('No cases selected')
    sources = [ROOT/'CMakeLists.txt', ROOT/'requirements.txt', ROOT/'docs/microduck_asset_manifest.json']
    for pattern in ('src/*.cpp','include/duck/*.hpp','scripts/*.py','tests/*.cpp'):
        sources.extend(ROOT.glob(pattern))
    source_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    report={'source_sha256':source_hashes,'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),'platform':platform.platform(),'python':platform.python_version(),'mujoco':mujoco.__version__,'numpy':np.__version__,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'cases':reports,'passed':all(r['passed'] for r in reports),'scope':'CPU FP64, one env, ground-only contact. Different constraint discretizations; throughput is not at matched converged accuracy.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
