#![feature(portable_simd)]
mod uloha9;

use std::fs::{create_dir_all};
use std::io::{BufWriter, Write};
use std::fs::File;
use std::path::Path;
use std::time::Instant;

fn main()
{
    let path_energy = Path::new("energies.txt");
    let file_energy = match File::create(&path_energy)
    {
        Err(why) => panic!("Error creating {}: {}", path_energy.display(), why),
        Ok(file) => file,
    };

    let path_absolute = Path::new("absolutes.txt");
    let file_absolute = match File::create(&path_absolute)
    {
        Err(why) => panic!("Error creating {}: {}", path_absolute.display(), why),
        Ok(file) => file,
    };

    let mut buf_writer_energy = BufWriter::new(file_energy);
    let mut buf_writer_absolute = BufWriter::new(file_absolute);

    const INIT_POS_PART1: [f64; 4] = [0.6, 0.0, 0.0, 0.03];
    let init_energy = uloha9::henon(INIT_POS_PART1);
    println!("Initial energy {}", init_energy);
    const EXP_DENOMINATOR: f64 = 15.0;
    const MAX_ITER: usize = 108;
    let final_short_dt_state = uloha9::rk4(INIT_POS_PART1, 1.0 / 10.0_f64.powf(MAX_ITER as f64 / EXP_DENOMINATOR), unsafe {10.0_f64.powf((MAX_ITER + 1) as f64 / EXP_DENOMINATOR).to_int_unchecked()});
    println!("RK4 - konvergence chyby");
    for i in 20..MAX_ITER
    {
        let before = Instant::now();
        let dt = 1.0 / 10.0_f64.powf(i as f64 / EXP_DENOMINATOR);
        let n_steps = unsafe { 10.0_f64.powf(i as f64 / EXP_DENOMINATOR).to_int_unchecked() };
        let final_state = uloha9::rk4(INIT_POS_PART1, dt, n_steps);
        let abs_dist = uloha9::l1_distance(final_state, final_short_dt_state);
        let abs_str = dt.to_string() + "," + &abs_dist.to_string() + "\n";
        buf_writer_absolute.write_all(abs_str.as_bytes()).unwrap();

        let final_energy = uloha9::henon(final_state);
        let energy_str = dt.to_string() + "," + &final_energy.to_string() + "\n";
        buf_writer_energy.write_all(energy_str.as_bytes()).unwrap();
        let elapsed = before.elapsed();
        println!("iter: {}, steps: {}, time elapsed: {:.3?}", i, n_steps, elapsed);
    }

    buf_writer_energy.flush().unwrap();
    buf_writer_absolute.flush().unwrap();

    println!("Poincaré (RK4)");
    //TODO: Adaptive step solver would be better
    const N_ORBITS: usize = 15;
    const BEST_DT: f64 = 5e-4;
    const N_STEPS: usize = 20000000;
    create_dir_all("poincare").unwrap();
    for k in 0..N_ORBITS + 1
    {
        let before = Instant::now();
        let x = 0.3 + (0.3 * (k as f64) / ((N_ORBITS + 2) as f64));
        let init_point = [x, 0.0, 0.0, 0.03];
        let final_pt = uloha9::rk4_poincare::<N_STEPS>(init_point, BEST_DT,&("poincare/poincare".to_owned() + &k.to_string() + ".txt"));
        let energy = uloha9::henon(final_pt);
        let init_energy = uloha9::henon(init_point);
        let elapsed = before.elapsed();
        println!("iter: {}, init energy: {}, final energy: {}, time elapsed: {:.3?}", k, init_energy, energy, elapsed);

    }

    println!("Single orbit (for 3D graph)");
    const INIT_POS_PART2: [f64; 4] = [0.35, 0.0, 0.0, 0.03];
    uloha9::rk4_verbose::<5000000>(INIT_POS_PART2, 1e-4, "orbit.txt");
}