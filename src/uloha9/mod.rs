use core_simd::{Simd};
use std::io::{BufWriter, Write};
use std::fs::File;
use std::path::Path;

const LANES: usize = 4;
type S = Simd<f64, LANES>;

const ZEROS: S = S::splat(0.0);
const HALVES_S: S = S::splat(1./2.);
const SIXTH_S: S = S::splat(1./6.);
const DOUBLE_S: S = S::splat(2.0);
const ONE_AND_HALF: S = S::splat(1.5);

pub fn henon(pt: [f64; 4]) -> f64
{
    (pt[2] * pt[2] + pt[3] * pt[3]) / 2.0 + (pt[0] * pt[0] + pt[1] * pt[1]) / 2.0 + (pt[0] * pt[0] * pt[1]) - (pt[1] * pt[1] * pt[1]) / 3.0
}

#[inline(always)]
fn deriv(pt_s: S) -> S
{
    let pt_pos = pt_s.to_array();
    let pxdot = - pt_pos[0] * (1.0 + 2.0 * pt_pos[1]);
    let pydot = - pt_pos[1] * (1.0 - pt_pos[1]) - pt_pos[0] * pt_pos[0];
    S::from_array([pt_pos[2], pt_pos[3], pxdot, pydot])
}

#[inline(always)]
pub fn rk4(pt_init: [f64; 4], dt: f64, n_steps: usize) -> [f64; 4]
{

    let dt_simd: S = S::splat(dt);
    let prefactor = dt_simd * SIXTH_S;
    let dt_half = HALVES_S * dt_simd;

    let mut pt_simd = S::from_array(pt_init);

    for _ in 0..n_steps
    {
        let k1_simd = deriv(pt_simd);
        let k2_simd = deriv(pt_simd + (dt_half * k1_simd));
        let k3_simd = deriv(pt_simd + (dt_half * k2_simd));
        let k4_simd = deriv(pt_simd + dt_simd * k3_simd);

        pt_simd += prefactor * (k1_simd + DOUBLE_S * (k2_simd + k3_simd) + k4_simd);
    }
    pt_simd.to_array()
}

pub fn rk4_verbose<const N_STEPS: usize>(pt_init: [f64; 4], dt: f64, log_name: &str) -> [f64; 4]
{

    // Prepare file for point logging
    let path = Path::new(log_name);
    let file = match File::create(&path)
    {
        Err(why) => panic!("Error creating {}: {}", path.display(), why),
        Ok(file) => file,
    };

    let mut buf_writer = BufWriter::new(file);

    let dt_simd: S = S::splat(dt);
    let prefactor = dt_simd * SIXTH_S;
    let dt_half = HALVES_S * dt_simd;

    let mut pt_simd = S::from_array(pt_init);
    for i in 0..N_STEPS
    {
        let pt = pt_simd.to_array();
        if i % 100 == 0 // log only every 100th step
        {
            let pt_str = (i as f64 * dt).to_string() + ", " + &pt[0].to_string() + ", " + &pt[1].to_string() + ", " + &pt[2].to_string() + ", " + &pt[3].to_string() + "\n";
            buf_writer.write_all(pt_str.as_bytes()).unwrap();
        }

        let k1_simd = deriv(pt_simd);
        let k2_simd = deriv(pt_simd + (dt_half * k1_simd));
        let k3_simd = deriv(pt_simd + (dt_half * k2_simd));
        let k4_simd = deriv(pt_simd + dt_simd * k3_simd);

        pt_simd += prefactor * (k1_simd + DOUBLE_S * (k2_simd + k3_simd) + k4_simd);

    }
    buf_writer.flush().unwrap();
    pt_simd.to_array()
}

#[inline(always)]
pub fn rk4_poincare<const N_STEPS: usize>(pt_init: [f64; 4], dt: f64, log_name: &str) -> [f64; 4]
{
    // Prepare file for point logging
    let path = Path::new(log_name);
    let file = match File::create(&path)
    {
        Err(why) => panic!("Error creating {}: {}", path.display(), why),
        Ok(file) => file,
    };

    let mut buf_writer = BufWriter::new(file);
    let mut buf_pt = [0.0; 16];
    for _ in 0..3
    {
        buf_pt = append_to_interpol_buffer(pt_init, buf_pt);
    }
    let dt_simd: S = S::splat(dt);

    let buffer_dt = [ZEROS, dt_simd, S::splat(2.0 * dt), S::splat(3.0 * dt)];
    let t0 = ONE_AND_HALF * dt_simd; //t0 for newton in interpolate and log method
    let prefactor = dt_simd * SIXTH_S;
    let dt_half = HALVES_S * dt_simd;

    let mut pt_simd = S::from_array(pt_init);

    for i in 0..N_STEPS
    {
        //Interpolate step (usually bw propagation is used, but I want to try something else!)
        let pt = pt_simd.to_array();
        buf_pt = append_to_interpol_buffer(pt, buf_pt);
        rk4_interpolate_0_and_log(dt * i as f64, buffer_dt, buf_pt, t0, &mut buf_writer);
        //RK4 Step
        let k1_simd = deriv(pt_simd);
        let k2_simd = deriv(pt_simd + (dt_half * k1_simd));
        let k3_simd = deriv(pt_simd + (dt_half * k2_simd));
        let k4_simd = deriv(pt_simd + dt_simd * k3_simd);

        pt_simd += prefactor * (k1_simd + DOUBLE_S * (k2_simd + k3_simd) + k4_simd);

    }
    buf_writer.flush().unwrap();
    pt_simd.to_array()
}

#[inline(always)]
pub fn rk4_interpolate_0_and_log(t: f64, xs: [S; 4], pt: [f64;16], t0: S, writer: &mut BufWriter<File>)
{
    //ys - 1, 5, 9, 13
    //first two y have same sign and last 2 ys have same sign (but opposite form the first pair)
    let k = ((pt[5] * pt[9]) < 0.0) && ((pt[1] * pt[5]) > 0.0) && ((pt[9] * pt[13]) > 0.0);
    if !k {return;}

    // Perform 1 step of Newton method (for x, we know derivative - x dot = p_x) and interpolate using aitken neville schema
    let p1 = S::from_slice(&pt[0..4]);
    let p2 = S::from_slice(&pt[4..8]);
    let p3 = S::from_slice(&pt[8..12]);
    let p4 = S::from_slice(&pt[12..16]);
    let pti = aitken_neville(p1,
                                 p2,
                                 p3,
                                 p4,
                                 xs,
                                 t0);
    let pti_arr = pti.to_array();
    let new_t = newton_step(t0.to_array()[0], pti_arr[1], pti_arr[3]); // t - y(t) / y'(t)
    let new_t_s = S::splat(new_t);
    let pti = aitken_neville(p1,
                         p2,
                         p3,
                         p4,
                         xs,
                         new_t_s);
    let pti_arr = pti.to_array();
    let new_t = newton_step(new_t, pti_arr[1], pti_arr[3]); // t - y(t) / y'(t)
    let new_t_s = S::splat(new_t);
    let pti = aitken_neville(p1,
                             p2,
                             p3,
                             p4,
                             xs,
                             new_t_s);

    let pts_final = pti.to_array();

    let buf = t.to_string() + "," + &pts_final[0].to_string() + ", " + &pts_final[2].to_string() + ", " + &pts_final[3].to_string() + "\n";
    writer.write_all(buf.as_bytes()).unwrap();

}

pub fn l1_distance(pt1: [f64; 4], pt2: [f64; 4]) -> f64
{
    let pt1_simd: S = S::from_array(pt1);
    let pt2_simd: S = S::from_array(pt2);
    let res = (pt1_simd - pt2_simd).abs();
    res.horizontal_sum()
}

#[inline(always)]
fn append_to_interpol_buffer(new_pt: [f64; 4], buf: [f64; 16]) -> [f64; 16]
{
    let mut k = [0.0; 16];
    k[..12].copy_from_slice(&buf[4..16]);
    k[12..16].copy_from_slice(&new_pt[..4]);
    k
}

#[inline(always)]
fn aitken_neville(pt1: S, pt2: S, pt3: S, pt4: S, xs: [S; 4], t: S) -> S
{
    let mut pts = [pt1, pt2, pt3, pt4];
    // Compute value in this point using Aitken-Neville
    for m in 1..4
    {
        for i in 0..(4-m) as usize
        {
            pts[i] = ((t - xs[i+m]) * pts[i] + (xs[i] - t) * pts[i+1]) / (xs[i] - xs[i+m])
        }
    }
    pts[0]
}

#[inline(always)]
fn newton_step(t: f64, x: f64, xprime: f64) -> f64
{
    t - x / xprime
}